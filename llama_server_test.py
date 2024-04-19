import requests

URL = 'http://localhost:8000/v1/chat/completions'
headers = {
	'accept': 'application/json',
	'Content-Type': 'application/json'
}
data = {
  "messages": [
    {
      "content": "you are a AI agent can do calculate and answer question in Chinese",
      "role": "system"
    },
    {
      "content": "the result of 1 plus 1?",
      "role": "user"
    },
	{
      "content": "3",
      "role": "assistant"
	},
    {
      "content": "1加1等于几?",
      "role": "user"
    }
  ]
}

response = requests.post(URL, headers=headers, json=data)
print(response.json())
print(response.json()['choices'][0]['message']['content'])
