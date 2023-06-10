import openai
import pyttsx3
import speech_recognition as sr
import time

# setup Open API key
openai.api_key = "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"

# initialize text to speech engine
engine = pyttsx3.init()


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        print("Skipping unknown error")


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_token=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response["choices"][0]["text"]


def speak_text(text):
    engine.say(text)


def main2():
    while True:
        # wait user to say "Genius" to start recording
        print("Say 'Genius' to start recording your question...")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            source.pause_threshold = 1
            audio = recognizer.listen(source)

            try:
                transcription = recognizer.recognize_google(source)
                if transcription.lower() == 'google':
                    # record audio
                    filename = "input.wav"
                    print("Say your Question")
                    with sr.Microphone() as q_source:
                        recognizer = sr.Recognizer()
                        q_source.pause_threshold = 1
                        audio = recognizer.listen(q_source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())

                    # Transcribe recorded audio to text
                    text = transcribe_audio_to_text(filename)
                    if text:
                        print(f"You said : {text}")
                        # generate response
                        response = generate_response(text)
                        print(f"GPT3 Response: {response}")

                        # Read the response with text to audio
                        speak_text(response)
            except Exception as e:
                print("An error occurred : {}", format(e))


def main():
    q = "10 names for dogs"
    print(f"GPT3 Question: {q}")
    response = generate_response(q)
    print(f"GPT3 Response: {response}")
    speak_text(response)


if __name__ == "__main__":
    main()
