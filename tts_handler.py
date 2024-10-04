import pyttsx3

def synthesize_text(text, out_file_name):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    
    voices = engine.getProperty("voices")
    
    engine.setProperty("voice", voices[1].id)
    engine.say(text=text)
    
    engine.save_to_file(text=text, filename=out_file_name)
    
    engine.runAndWait()
