import speech_recognition as sr

# Create a recognizer instance
r = sr.Recognizer()
r.energy_threshold = 1000


# Function to listen to audio from microphone and perform speech recognition
def listen_for_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        # Perform speech recognition
        text = r.recognize_sphinx(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


# Main loop
while True:
    # Listen for speech
    speech = listen_for_speech()

    if speech:
        print("You said:", speech)
