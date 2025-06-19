async function speakText() {
  const text = document.getElementById("text").value;
  const language = document.getElementById("lang").value;

  const response = await fetch("http://localhost:5000/speak", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text, language })
  });

  const blob = await response.blob();
  const audioUrl = URL.createObjectURL(blob);

  const audio = new Audio(audioUrl);
  audio.play();
}
