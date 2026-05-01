from gtts import gTTS

# Create a sample voice command for your agent
## Test Tamil in your Agent
#message = HumanMessage(content="இந்த மருந்து அட்டையில் எத்தனை மாத்திரைகள் உள்ளன? (How many tablets are in this strip?)")
text = "இந்த மருந்து அட்டையில் எத்தனை மாத்திரைகள் உள்ளன?(Hola!)"
tts = gTTS(text=text, lang='en')
tts.save("voice_note.wav")

print("Test file 'voice_note.wav' created successfully!")