// document.getElementById("genRandomCapcha").addEventListener("click", function() {
//     fetch("/randomCapcha")
//         .then(response => response.json())
//         .then(data => {
//             if (data.status) {
//                 imgElement = document.getElementById("capcha_img");
//                 imgElement.src = data.path + '?timestamp=' + new Date().getTime(); // Adding timestamp to avoid caching
//                 imgElement.style.display = "block";
//             } else{
//                 console.error("Error:",'There was a error while fetching values from the api');
//             }
//         })
//         .catch(error => console.error('Error:', error));
// });


// document.getElementById('customCapcha').addEventListener('click',function(){
//     const userInput = document.getElementById("capcha_input").value;

//     if (userInput.length === 4) { // Ensure the input is 4 digits
//         fetch('/generateCapcha', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({ capcha: userInput }) // Send the input as JSON
//         })
//         .then(response => response.json())
//         .then(data => {
//             if (data.status) {
//                 imgElement = document.getElementById("custom_capcha");
//                 imgElement.src = data.path + '?timestamp=' + new Date().getTime(); // Adding timestamp to avoid caching
//                 imgElement.style.display = "block";
//             } else{
//                 console.error("Error:",'There was a error while fetching values from the api');
//             }
//         })
//         .catch(error => console.error('Error:', error));
//     }

// })

document.getElementById("uploadCapchaBtn").addEventListener("click", function () {
    const fileInput = document.getElementById("myFile");
    const outputContainer = document.getElementById("ouptutDiv");
    const output = document.getElementById("output");
    const formData = new FormData();
    const originalImg = document.getElementById("capchaImg");
    const loader = document.getElementById("loaderOverlay");
    const Count = document.getElementById("characters").value;

    // Initialize the loader
    loader.style.display = "flex"

    if (fileInput.files.length > 0) {
        formData.append("file", fileInput.files[0]);
        formData.append("count",Count)
    }

    // Make the API call to /uploadCapcha
    fetch("/uploadCapcha", {
        method: "POST",
        body: formData,
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Error in prediction API call.");
            }
            return response.json(); // Return the response as text
        })
        .then((result) => {
            loader.style.display = "none"
            // Display the prediction result
            output.innerHTML = `${result.data}`;
            outputContainer.style.display = "flex"; // Show the div

            // Displaying the uploaded image
            const uploadedImageURL = URL.createObjectURL(fileInput.files[0]);
            originalImg.src = uploadedImageURL;
        })
        .catch((error) => {
            loader.style.display = "none"
            output.innerHTML = "Can't Process the image";
            outputContainer.style.display = "flex";

            // Displaying the uploaded image
            const uploadedImageURL = URL.createObjectURL(fileInput.files[0]);
            originalImg.src = uploadedImageURL;

        });
});



