import cv2
from config import audioCollection, frameCollection, questionCollection
import os
from const import UPLOAD_DIRECTORY
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import librosa
import httpx
import speech_recognition as sr
import math
from config import modelHand, faceCascade, recognizer, mpDrawing, mpHands, bucket, model
import io
from google.cloud import speech_v1p1beta1 as speech
import json
import wave
from pydub import AudioSegment
from PIL import Image
from tensorflow.keras.models import load_model

emotionModel = load_model(os.getenv("EMOTION_MODEL"))

emotionMapping = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

async def analyze(video_location, email, uuid):
    try:
        question = questionCollection.find_one({"id": uuid})

        print("read video start")
        video_capture = cv2.VideoCapture(video_location)
        audio_location = UPLOAD_DIRECTORY / f"{uuid}.wav"
        videoFile = VideoFileClip(str(video_location))
        audioFile = videoFile.audio
        if audioFile == None:
            print("Your video doesn't have audio")
            raise Exception("Your video doesn't have audio")
        audioFile.write_audiofile(str(audio_location))
        audioFile.close()
        videoFile.close()

        frames = []

        if not video_capture.isOpened():
            raise Exception("failed to read video")

        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print("read video done")
        messages = question["messages"]
        messages.append("read video done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print("frame start")
        frame_count = 0
        while True:
            success, frame = video_capture.read()

            if not success:
                break 

            frames.append(frame)

            frame_count += 1

        video_capture.release()


        print("frame done")
        messages = question["messages"]
        messages.append("frame done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print(f"frame length: {len(frames)}")
        print("emotions start")
        emotions, frames, faceWidth = await analyze_emotions(frames, fps)
        print(emotions)
        print("emotions done")
        messages = question["messages"]
        messages.append("emotions done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print("hand start")
        handsResult, frames = await analyze_hands(frames, frame_width, frame_height)
        print("hand done")
        messages = question["messages"]
        messages.append("hand done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print("upload gcs start")
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        new_video_location = str(video_location).split(".")[0] + ".avi"
        out = cv2.VideoWriter(new_video_location, fourcc, fps, (frame_width, frame_height))
        for frame in frames:
            out.write(frame)
        out.release()
        
        newVideo = VideoFileClip(str(new_video_location))
        newAudio = AudioFileClip(str(audio_location)).set_duration(newVideo.duration)
        newVideo = newVideo.set_audio(newAudio)
        file_extension = os.path.splitext(video_location)[-1].lower()

        if file_extension == ".mp4":
            codec = 'libx264'
            audio_codec = 'aac'
        elif file_extension == ".webm":
            codec = 'libvpx'
            audio_codec = 'libvorbis'
        else:
            raise ValueError(f"Unsupported video format: {file_extension}")
        newVideo.write_videofile(str(video_location), codec=codec, audio_codec=audio_codec)
        newVideo.close()
        newAudio.close()
        blob = bucket.blob(str(video_location))
        blob.upload_from_filename(str(video_location))
        print("upload gcs done")
        messages = question["messages"]
        messages.append("gcs upload done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print("translate start")
        resultTranslate, text = translate(str(audio_location), get_sample_rate(str(audio_location)))
        print("translate done")
        messages = question["messages"]
        messages.append("translate done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})


        print("snr start")
        audio, ser = librosa.load(str(audio_location), sr=None)
        snr_values = compute_snr(audio, ser)
        arr_cleaned = np.nan_to_num(snr_values, nan=0.0)
        print("snr done")
        messages = question["messages"]
        messages.append("snr done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print("calculation start")
        for index, phrase in enumerate(resultTranslate):
            startIndex = math.floor(phrase["start_time"]*fps)
            endIndex = math.floor(phrase["end_time"]*fps)
            movement = 0
            for i in range(startIndex, endIndex):
                movement += handsResult[i]
            if(movement > faceWidth):
                resultTranslate[index]["actual_gesture"] = True
            else:
                resultTranslate[index]["actual_gesture"] = False

        for index, phrase in enumerate(resultTranslate):
            startIndex = math.floor(phrase["start_time"]*fps)
            endIndex = math.floor(phrase["end_time"]*fps)
            actualIndex = (startIndex+endIndex)//2
            print(actualIndex, emotions[actualIndex])
            resultTranslate[index]["actual_emotion"] = emotions[actualIndex]

        print(resultTranslate)


        print("calculation done")
        messages = question["messages"]
        messages.append("calculation done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print("mongo function")
        frameCollection.insert_one({"id": uuid, "email": email, "emotions": emotions, "hands": handsResult})
        audioCollection.insert_one({"id": uuid, "email": email, "snr": arr_cleaned.tolist(), "answer": text})
        messages = question["messages"]
        messages.append("all done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages, "status": "SUCCESS", "answer": text, "result": resultTranslate}})


        os.remove(video_location)
        os.remove(new_video_location)
        os.remove(audio_location)
        print("all done")
    except Exception as e:
        print(e)
        messages = question["messages"]
        messages.append(str(e))
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages, "status": "ERROR"}})

async def analyze_emotions(frames, fps):
    faceWidth = 0
    emotions = []
    currEmotion = ""
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            faceWidth = w
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cropped = frame[y:y+h, x:x+w]

            if(index % (fps//5) == 0):
                # async with httpx.AsyncClient() as client:
                #     response = await client.post(os.getenv('API_EMOTION_URL'), json={ "matrix": cropped.tolist()})
                #     response.raise_for_status()
                #     data = response.json()
                #     currEmotion = data["prediction"]
                currEmotion = predict(cropped)
                print(index, currEmotion)
            emotions.append(currEmotion)
        else:
            emotions.append("unknown")
    
    return emotions, frames, faceWidth

async def analyze_hands(frames, width, height):
    handsPositionX = []
    handsPositionY = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = modelHand.process(rgb_frame)
        if hands.multi_hand_landmarks:
            first_hand = hands.multi_hand_landmarks[0]
            if len(hands.multi_hand_landmarks) == 1:
                handsPositionX.append(first_hand.landmark[0].x*width)
                handsPositionY.append(first_hand.landmark[0].y*height)
            else:
                second_hand = hands.multi_hand_landmarks[1]
                handsPositionX.append((first_hand.landmark[0].x*width + second_hand.landmark[0].x*width)/2)
                handsPositionY.append((first_hand.landmark[0].y*height + second_hand.landmark[0].y*height)/2)

            for hand_landmarks in hands.multi_hand_landmarks:
                mpDrawing.draw_landmarks(
                    frame, 
                    hand_landmarks,
                    mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mpDrawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        else:
            handsPositionX.append(-1)
            handsPositionY.append(-1)
    handsPositionXnp = np.array(handsPositionX)
    handsPositionXnp = np.diff(handsPositionXnp)
    handsPositionYnp = np.array(handsPositionY)
    handsPositionYnp = np.diff(handsPositionYnp)
    handsPositionXnp = handsPositionXnp.tolist()
    handsPositionYnp = handsPositionYnp.tolist()
    handsResult = []
    for i in range(len(handsPositionXnp)):
        handsResult.append(math.sqrt(handsPositionXnp[i]**2 + handsPositionYnp[i]**2))

    return handsResult, frames

def translate(speech_file, sample_rate):
    client = speech.SpeechClient()
    
    audio = AudioSegment.from_wav(speech_file)
    mono_audio = audio.set_channels(1)
    mono_audio.export(f"final-{speech_file}", format="wav")

    with io.open(f"final-{speech_file}", "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="id-ID",
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=500)
    prompt = """
                Split the following text by phrases. Provide an emotion angry/disgust/fear/happy/neutral/sad/surprise for each phrase and indicate whether hand gestures are needed when delivering it. 
                Here is the text: {transcript}. 
                Provide the format as an array: [{{"phrase": "text", "emotion": "angry/disgust/fear/happy/neutral/sad/surprise", "gesture": true/false}}]. 
                Return it as an array without any additional characters or formatting such as ```json``` or backticks.
            """
    
    formatted = prompt.format(transcript=response.results[0].alternatives[0].transcript)
    resGemini = model.generate_content([formatted])
    os.remove(f"final-{speech_file}")

    object = json.loads(resGemini.text)

    phraseIndex = 0
    for word_info in response.results[0].alternatives[0].words:
        word = word_info.word
        start_time = word_info.start_time.total_seconds()
        end_time = word_info.end_time.total_seconds()
        for i in range(phraseIndex, len(object)):
            if word.lower() in object[phraseIndex]["phrase"].lower():
                if "start_time" not in object[phraseIndex]:
                    object[phraseIndex]["start_time"] = start_time
                object[phraseIndex]["end_time"] = end_time
                break
            else:
                phraseIndex += 1
    return object, response.results[0].alternatives[0].transcript

def compute_snr(audio, sr, frame_length=1024, hop_length=512):
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    power_spec = np.abs(stft)**2
    signal_power = np.mean(power_spec, axis=0)
    noise_power = np.median(power_spec, axis=0)
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def get_sample_rate(wav_file):
    with wave.open(wav_file, 'rb') as wav:
        sample_rate = wav.getframerate()
        return sample_rate
    
def preprocess(image_array):
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image_matrix):
    image = Image.fromarray(image_matrix)

    image = image.resize((48, 48))

    resized_array = np.array(image)

    processed_image = preprocess(resized_array)
    prediction = emotionModel.predict(processed_image)
    predicted_class_index = int(np.argmax(prediction, axis=1)[0])
    return emotionMapping.get(predicted_class_index, "unknown")

async def analyzeCode(video_location, uuid, code):
    try:
        question = questionCollection.find_one({"id": uuid})

        print("read video start")
        video_capture = cv2.VideoCapture(video_location)
        audio_location = UPLOAD_DIRECTORY / f"{uuid}.wav"
        videoFile = VideoFileClip(str(video_location))
        audioFile = videoFile.audio
        if audioFile == None:
            print("Your video doesn't have audio")
            raise Exception("Your video doesn't have audio")
        audioFile.write_audiofile(str(audio_location))
        audioFile.close()
        videoFile.close()

        print("translate start")
        resultTranslate, text = translate(str(audio_location), get_sample_rate(str(audio_location)))
        print("translate done")
        messages = question["messages"]
        messages.append("translate done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages}})

        print("mongo function")
        messages = question["messages"]
        messages.append("all done")
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages, "status": "SUCCESS", "answer": text, "code": code, "result": resultTranslate}})


        os.remove(video_location)
        os.remove(audio_location)
        print("all done")
    except Exception as e:
        print(e)
        messages = question["messages"]
        messages.append(str(e))
        question["messages"] = messages        
        questionCollection.find_one_and_update({"id": uuid},{ "$set": { "messages": messages, "status": "ERROR"}})