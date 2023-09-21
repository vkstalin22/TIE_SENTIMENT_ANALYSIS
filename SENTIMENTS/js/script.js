document.addEventListener("DOMContentLoaded", function () {
    const analyzeBtn = document.getElementById("analyzeBtn");
    const keywordInput = document.getElementById("keyword");
    const resultParagraph = document.getElementById("result");

    analyzeBtn.addEventListener("click", () => {
        const keyword = keywordInput.value;

        if (!keyword) {
            alert("Please enter a keyword.");
            return;
        }

        // Send an API request to your Flask server for sentiment analysis
        fetch(`/analyze?keyword=${encodeURIComponent(keyword)}`)
            .then((response) => response.json())
            .then((data) => {
                resultParagraph.innerText = `Sentiment of '${keyword}': ${data.sentiment}`;
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("An error occurred while analyzing sentiment.");
            });
    });
});
