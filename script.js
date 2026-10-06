// ============================================================
// TEXTILEVISION - FRONTEND SCRIPT
// ============================================================


// ============================================================
// ELEMENTS
// ============================================================

const dropZone =
    document.getElementById("dropZone");

const imageInput =
    document.getElementById("imageInput");

const fileName =
    document.getElementById("fileName");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const imageInfo =
    document.getElementById("imageInfo");

const removeImage =
    document.getElementById("removeImage");

const analyzeButton =
    document.getElementById("analyzeButton");


// ============================================================
// RESULT ELEMENTS
// ============================================================

const resultsSection =
    document.getElementById("resultsSection");

const predictionBadge =
    document.getElementById("predictionBadge");

const cnnPrediction =
    document.getElementById("cnnPrediction");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceBar =
    document.getElementById("confidenceBar");

const svmPrediction =
    document.getElementById("svmPrediction");


// FINAL STATUS CARD

const inspectionStatus =
    document.getElementById("inspectionStatus");

const inspectionStatusCard =
    document.getElementById("inspectionStatusCard");


// RESULT IMAGES

const originalResultImage =
    document.getElementById("originalResultImage");

const gradcamImage =
    document.getElementById("gradcamImage");

const segmentationImage =
    document.getElementById("segmentationImage");


// SEGMENTATION METRICS

const iouValue =
    document.getElementById("iouValue");

const diceValue =
    document.getElementById("diceValue");

const defectArea =
    document.getElementById("defectArea");


// TABLE ELEMENTS

const tableCNN =
    document.getElementById("tableCNN");

const tableConfidence =
    document.getElementById("tableConfidence");

const tableCNNStatus =
    document.getElementById("tableCNNStatus");


const tableSVM =
    document.getElementById("tableSVM");

const tableSVMConfidence =
    document.getElementById("tableSVMConfidence");

const tableSVMStatus =
    document.getElementById("tableSVMStatus");


const tableDice =
    document.getElementById("tableDice");

const tableUnetStatus =
    document.getElementById("tableUnetStatus");


// ============================================================
// APPLICATION STATE
// ============================================================

let selectedFile = null;


// ============================================================
// FILE INPUT
// ============================================================

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];

        if (!file) {
            return;
        }

        handleSelectedFile(file);
    }
);


// ============================================================
// HANDLE SELECTED FILE
// ============================================================

function handleSelectedFile(file) {

    // --------------------------------------------------------
    // CHECK IMAGE TYPE
    // --------------------------------------------------------

    if (!file.type.startsWith("image/")) {

        alert(
            "Please select a valid image file."
        );

        return;
    }


    // --------------------------------------------------------
    // SAVE FILE
    // --------------------------------------------------------

    selectedFile = file;


    console.log(
        "Selected file:",
        file.name
    );

    console.log(
        "File type:",
        file.type
    );

    console.log(
        "File size:",
        file.size
    );


    // --------------------------------------------------------
    // DISPLAY FILE NAME
    // --------------------------------------------------------

    fileName.textContent =
        file.name;


    // --------------------------------------------------------
    // IMAGE PREVIEW
    // --------------------------------------------------------

    const imageURL =
        URL.createObjectURL(file);

    imagePreview.src =
        imageURL;

    imagePreview.style.display =
        "block";


    previewContainer.classList.remove(
        "hidden"
    );


    // --------------------------------------------------------
    // IMAGE INFORMATION
    // --------------------------------------------------------

    imageInfo.textContent =
        `${file.type} • ${(file.size / 1024).toFixed(1)} KB`;


    // --------------------------------------------------------
    // ENABLE ANALYZE BUTTON
    // --------------------------------------------------------

    analyzeButton.disabled =
        false;


    // --------------------------------------------------------
    // HIDE OLD RESULTS
    // --------------------------------------------------------

    resultsSection.classList.add(
        "hidden"
    );


    console.log(
        "Image preview displayed."
    );
}


// ============================================================
// REMOVE IMAGE
// ============================================================

removeImage.addEventListener(
    "click",
    function () {

        selectedFile = null;

        imageInput.value = "";

        imagePreview.src = "";

        imagePreview.style.display =
            "none";

        previewContainer.classList.add(
            "hidden"
        );

        fileName.textContent =
            "No image selected";

        imageInfo.textContent =
            "Ready for analysis";

        analyzeButton.disabled =
            true;

        resultsSection.classList.add(
            "hidden"
        );


        console.log(
            "Image removed."
        );
    }
);


// ============================================================
// DRAG OVER
// ============================================================

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.classList.add(
            "drag-over"
        );
    }
);


// ============================================================
// DRAG LEAVE
// ============================================================

dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.classList.remove(
            "drag-over"
        );
    }
);


// ============================================================
// DROP IMAGE
// ============================================================

dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.classList.remove(
            "drag-over"
        );


        const files =
            event.dataTransfer.files;


        if (
            !files ||
            files.length === 0
        ) {

            return;
        }


        const file =
            files[0];


        handleSelectedFile(file);
    }
);


// ============================================================
// DISPLAY PREDICTION
// ============================================================

function displayPrediction(data) {

    console.log(
        "==================================="
    );

    console.log(
        "Displaying prediction..."
    );

    console.log(
        "Backend response:",
        data
    );


    // ========================================================
    // CHECK RESPONSE
    // ========================================================

    if (
        !data ||
        !data.prediction
    ) {

        console.error(
            "Invalid backend response:",
            data
        );

        alert(
            "Invalid prediction response from backend."
        );

        return;
    }


    // ========================================================
    // GET PREDICTION OBJECT
    // ========================================================

    const prediction =
        data.prediction;


    // ========================================================
    // CNN DATA
    // ========================================================

    const cnnClass =
        prediction.cnn_class || "UNKNOWN";


    const cnnConfidence =
        Number(
            prediction.confidence || 0
        );


    const cnnRawPrediction =
        Number(
            prediction.raw_prediction || 0
        );


    // ========================================================
    // SVM DATA
    // ========================================================

    const svmClass =
        prediction.svm_class || "UNKNOWN";


    const svmConfidence =
        Number(
            prediction.svm_confidence || 0
        );


    // ========================================================
    // FINAL STATUS
    // ========================================================

    let finalStatus =
        prediction.final_status;


    // --------------------------------------------------------
    // BACKWARD COMPATIBILITY
    // --------------------------------------------------------

    if (!finalStatus) {

        finalStatus =
            prediction.inspection_status;
    }


    if (!finalStatus) {

        finalStatus =
            "REVIEW";
    }


    // ========================================================
    // LOG RESULTS
    // ========================================================

    console.log(
        "CNN class:",
        cnnClass
    );

    console.log(
        "CNN confidence:",
        cnnConfidence
    );

    console.log(
        "CNN raw prediction:",
        cnnRawPrediction
    );

    console.log(
        "SVM class:",
        svmClass
    );

    console.log(
        "SVM confidence:",
        svmConfidence
    );

    console.log(
        "Final status:",
        finalStatus
    );

    console.log(
        "Grad-CAM URL:",
        prediction.gradcam_url
    );

    console.log(
        "Segmentation URL:",
        prediction.segmentation_url
    );

    console.log(
        "Defect area:",
        prediction.defect_area
    );


    // ========================================================
    // FINAL DECISION BADGE
    // ========================================================

    predictionBadge.textContent =
        finalStatus;


    // ========================================================
    // FINAL INSPECTION STATUS
    // ========================================================

    inspectionStatus.textContent =
        finalStatus;


    // The second status card was added to the HTML.

    if (inspectionStatusCard) {

        inspectionStatusCard.textContent =
            finalStatus;
    }


    // ========================================================
    // CNN CLASSIFICATION
    // ========================================================

    cnnPrediction.textContent =
        cnnClass;


    // ========================================================
    // CNN CONFIDENCE
    // ========================================================

    confidenceValue.textContent =
        `${cnnConfidence.toFixed(2)}%`;


    confidenceBar.style.width =
        `${Math.min(cnnConfidence, 100)}%`;


    // ========================================================
    // SVM CLASSIFICATION
    // ========================================================

    if (
        prediction.svm_class
    ) {

        svmPrediction.textContent =
            `${svmClass} (${svmConfidence.toFixed(2)}%)`;

    } else {

        svmPrediction.textContent =
            "Not available";
    }


    // ========================================================
    // CNN TABLE
    // ========================================================

    tableCNN.textContent =
        cnnClass;


    tableConfidence.textContent =
        `${cnnConfidence.toFixed(2)}%`;


    // ========================================================
    // CNN TABLE STATUS
    // ========================================================

    if (tableCNNStatus) {

        tableCNNStatus.textContent =
            cnnClass;
    }


    // ========================================================
    // SVM TABLE
    // ========================================================

    if (
        prediction.svm_class
    ) {

        tableSVM.textContent =
            svmClass;

    } else {

        tableSVM.textContent =
            "Not available";
    }


    // ========================================================
    // SVM TABLE CONFIDENCE
    // ========================================================

    if (
        tableSVMConfidence
    ) {

        if (
            prediction.svm_class
        ) {

            tableSVMConfidence.textContent =
                `${svmConfidence.toFixed(2)}%`;

        } else {

            tableSVMConfidence.textContent =
                "N/A";
        }
    }


    // ========================================================
    // SVM TABLE STATUS
    // ========================================================

    if (tableSVMStatus) {

        if (
            prediction.svm_class
        ) {

            tableSVMStatus.textContent =
                svmClass;

        } else {

            tableSVMStatus.textContent =
                "Unavailable";
        }
    }


    // ========================================================
    // ORIGINAL IMAGE
    // ========================================================

    if (
        selectedFile &&
        originalResultImage
    ) {

        const originalURL =
            URL.createObjectURL(
                selectedFile
            );


        originalResultImage.src =
            originalURL;


        originalResultImage.style.display =
            "block";


        console.log(
            "Original result image displayed."
        );
    }


    // ========================================================
    // GRAD-CAM
    // ========================================================

    if (
        prediction.gradcam_url &&
        gradcamImage
    ) {

        gradcamImage.src =
            prediction.gradcam_url +
            "?t=" +
            Date.now();


        gradcamImage.style.display =
            "block";


        console.log(
            "Grad-CAM image displayed:",
            prediction.gradcam_url
        );

    } else if (
        gradcamImage
    ) {

        gradcamImage.src =
            "";

        gradcamImage.style.display =
            "none";
    }


    // ========================================================
    // U-NET SEGMENTATION
    // ========================================================

    if (
        prediction.segmentation_url &&
        segmentationImage
    ) {

        segmentationImage.src =
            prediction.segmentation_url +
            "?t=" +
            Date.now();


        segmentationImage.style.display =
            "block";


        console.log(
            "U-Net segmentation displayed:",
            prediction.segmentation_url
        );


        // ----------------------------------------------------
        // U-NET TABLE STATUS
        // ----------------------------------------------------

        if (tableUnetStatus) {

            tableUnetStatus.textContent =
                "Generated";
        }

    } else if (
        segmentationImage
    ) {

        segmentationImage.src =
            "";

        segmentationImage.style.display =
            "none";


        console.log(
            "No U-Net segmentation available."
        );


        if (tableUnetStatus) {

            tableUnetStatus.textContent =
                "Not generated";
        }
    }


    // ========================================================
    // DEFECT AREA
    // ========================================================

    if (defectArea) {

        if (
            prediction.defect_area !== undefined &&
            prediction.defect_area !== null
        ) {

            defectArea.textContent =
                `${Number(
                    prediction.defect_area
                ).toFixed(2)}%`;

        } else {

            defectArea.textContent =
                "N/A";
        }
    }


    // ========================================================
    // IoU
    // ========================================================

    if (iouValue) {

        // Live uploaded images do not have ground-truth masks.

        iouValue.textContent =
            "N/A";
    }


    // ========================================================
    // DICE
    // ========================================================

    if (diceValue) {

        // Live uploaded images do not have ground-truth masks.

        diceValue.textContent =
            "N/A";
    }


    // ========================================================
    // DICE TABLE
    // ========================================================

    if (tableDice) {

        tableDice.textContent =
            "N/A";
    }


    // ========================================================
    // SHOW RESULTS
    // ========================================================

    resultsSection.classList.remove(
        "hidden"
    );


    // ========================================================
    // SCROLL TO RESULTS
    // ========================================================

    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });


    console.log(
        "==================================="
    );

    console.log(
        "Prediction display complete."
    );

    console.log(
        "==================================="
    );
}


// ============================================================
// ANALYZE BUTTON
// ============================================================

analyzeButton.addEventListener(
    "click",
    async function () {


        // ====================================================
        // CHECK FILE
        // ====================================================

        if (!selectedFile) {

            alert(
                "Please select an image first."
            );

            return;
        }


        console.log(
            "==================================="
        );

        console.log(
            "Sending image to Flask..."
        );

        console.log(
            "File:",
            selectedFile.name
        );

        console.log(
            "==================================="
        );


        // ====================================================
        // FORM DATA
        // ====================================================

        const formData =
            new FormData();


        formData.append(
            "image",
            selectedFile
        );


        // ====================================================
        // ANALYZING STATE
        // ====================================================

        analyzeButton.disabled =
            true;

        analyzeButton.textContent =
            "Analyzing...";


        try {


            // =================================================
            // SEND REQUEST
            // =================================================

            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            console.log(
                "Backend HTTP status:",
                response.status
            );


            // =================================================
            // READ JSON
            // =================================================

            const data =
                await response.json();


            console.log(
                "Backend response:",
                data
            );


            // =================================================
            // CHECK RESPONSE
            // =================================================

            if (!response.ok) {

                throw new Error(
                    data.message ||
                    "Image analysis failed."
                );
            }


            if (
                data.status !== "success"
            ) {

                throw new Error(
                    data.message ||
                    "Backend prediction failed."
                );
            }


            // =================================================
            // DISPLAY RESULT
            // =================================================

            displayPrediction(
                data
            );


        } catch (error) {


            // =================================================
            // ERROR HANDLING
            // =================================================

            console.error(
                "Analysis error:",
                error
            );


            alert(
                "Image analysis failed.\n\n" +
                error.message
            );


        } finally {


            // =================================================
            // RESTORE BUTTON
            // =================================================

            analyzeButton.disabled =
                false;

            analyzeButton.textContent =
                "Analyze Fabric";
        }
    }
);