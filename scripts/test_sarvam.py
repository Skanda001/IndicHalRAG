from sarvamai import SarvamAI
import os
from dotenv import load_dotenv

load_dotenv()
client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

response = client.chat.completions(
       messages=[{"role": "user", "content": "ಭಾರತದ ಅತಿ ಎತ್ತರದ ಶಿಖರ ಯಾವುದು?"}],
       model="sarvam-105b",
       max_tokens=100,
       reasoning_effort=None,   # important — see note below
   )
print(response.choices[0].message.content)
