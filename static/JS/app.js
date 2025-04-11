document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const previewImage = document.getElementById("previewImage");
  const overlayCanvas = document.getElementById("overlayCanvas");
  const analysisSection = document.getElementById("analysisSection");
  const generateReportBtn = document.getElementById("generateReportBtn");
  const abcdInfoBtn = document.getElementById("abcdInfoBtn");
  const abcdModal = document.getElementById("abcdModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const visualizationMode = document.getElementById("visualizationMode");
  const toggleOverlayBtn = document.getElementById("toggleOverlayBtn");

  // Score elements
  const asymmetryScoreEl = document.getElementById("asymmetryScore");
  const borderScoreEl = document.getElementById("borderScore");
  const colorScoreEl = document.getElementById("colorScore");
  const diameterScoreEl = document.getElementById("diameterScore");
  const tdsScoreEl = document.getElementById("tdsScore");
  const interpretationTextEl = document.getElementById("interpretationText");

  // Zoom controls
  const zoomInBtn = document.getElementById("zoomInBtn");
  const zoomOutBtn = document.getElementById("zoomOutBtn");
  const resetZoomBtn = document.getElementById("resetZoomBtn");

  // App state
  let currentImage = null;
  let currentScale = 1;
  let analysisResults = null;
  let tdsChart = null;
  let showOverlay = true;

  // Event Listeners
  uploadBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileSelect);

  // Drag and drop events
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");

    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelect({ target: fileInput });
    }
  });

  // Zoom controls
  zoomInBtn.addEventListener("click", () => adjustZoom(1.2));
  zoomOutBtn.addEventListener("click", () => adjustZoom(0.8));
  resetZoomBtn.addEventListener("click", resetZoom);

  // Generate report
  generateReportBtn.addEventListener("click", generateReport);

  // ABCD Info Modal
  abcdInfoBtn.addEventListener("click", () => {
    abcdModal.classList.add("active");
    document.body.style.overflow = "hidden";
  });

  closeModalBtn.addEventListener("click", () => {
    abcdModal.classList.remove("active");
    document.body.style.overflow = "";
  });

  // Visualization mode
  visualizationMode.addEventListener("change", drawOverlay);
  toggleOverlayBtn.addEventListener("click", () => {
    showOverlay = !showOverlay;
    drawOverlay();
    toggleOverlayBtn.querySelector("span").textContent = showOverlay
      ? "Hide Overlay"
      : "Show Overlay";
  });

  // Close modal when clicking outside
  abcdModal.addEventListener("click", (e) => {
    if (e.target === abcdModal) {
      abcdModal.classList.remove("active");
      document.body.style.overflow = "";
    }
  });

  // Functions
  function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file || !file.type.match("image.*")) {
      alert("Please upload a valid image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/segment", {
      method: "POST",
      body: formData,
    })
      .then((res) => {
        if (!res.ok) throw new Error("Segmentation failed.");
        return res.json();
      })
      .then((data) => {
        const originalImage = document.getElementById("originalImage");
        originalImage.src = "data:image/png;base64," + data.original_image;
        originalImage.style.display = "block";

        const previewImage = document.getElementById("previewImage");
        previewImage.src = "data:image/png;base64," + data.segmented_image;
        previewImage.style.display = "block";

        const analysisSection = document.getElementById("analysisSection");
        analysisSection.style.display = "block";

        // Update scores
        document.getElementById("asymmetryScore").textContent = data.asymmetry;
        document.getElementById("borderScore").textContent = data.border;
        document.getElementById("diameterScore").textContent =
          data.diameter + " px";
        document.getElementById("colorScore").textContent = data.color_score;

        // ✅ Correct TDS calculation using normalized diameter score
        const A = data.asymmetry;
        const B = data.border;
        const C = data.color_score;
        const D = data.diameter_score; // Already normalized

        const TDS = 1.3 * A + 0.1 * C + 0.5 * B + 0.5 * D * 20;

        const tdsScoreEl = document.getElementById("tdsScore");
        tdsScoreEl.textContent = TDS.toFixed(2);

        // Update color card
        const colorLabelList = data.present_colors.join(", ");
        const colorCard = document.getElementById("colorCard");
        colorCard.querySelector(".score-details").innerHTML = `
            <p>Colors: <span>${colorLabelList}</span></p>
            <p>Weight: <span>×0.5</span></p>
        `;

        setTimeout(() => {
          analysisSection.scrollIntoView({ behavior: "smooth" });
        }, 100);
      })
      .catch((err) => {
        console.error("❌ Error:", err);
        alert("Something went wrong during segmentation.");
      });
  }

  function adjustZoom(factor) {
    currentScale *= factor;
    previewImage.style.transform = `translate(-50%, -50%) scale(${currentScale})`;
    drawOverlay();
  }

  function resetZoom() {
    currentScale = 1;
    previewImage.style.transform = "translate(-50%, -50%) scale(1)";
    drawOverlay();
  }

  function analyzeImage(imageFile) {
    // Show loading state
    asymmetryScoreEl.textContent = "...";
    borderScoreEl.textContent = "...";
    colorScoreEl.textContent = "...";
    diameterScoreEl.textContent = "...";
    tdsScoreEl.textContent = "...";
    interpretationTextEl.textContent = "Analyzing image...";
    interpretationTextEl.className = "interpretation";
    generateReportBtn.disabled = true;

    // In a real implementation, this would call your backend API
    // For demo purposes, we'll simulate an API call with timeout
    simulateApiCall(imageFile);
  }

  function simulateApiCall(imageFile) {
    // This is where you would normally send the image to your backend
    // For now, we'll use mock data after a delay

    setTimeout(() => {
      // Mock data - replace with actual API response
      analysisResults = {
        asymmetry: 2,
        border: 6,
        colors: 4,
        diameter: 3.2,
        asymmetryDetails: {
          axes: [
            { x1: 0.2, y1: 0.5, x2: 0.8, y2: 0.5 }, // horizontal
            { x1: 0.5, y1: 0.2, x2: 0.5, y2: 0.8 }, // vertical
          ],
          asymmetric: [true, true], // both axes asymmetric
        },
        borderDetails: {
          segments: [1, 3, 4, 5, 7, 8], // abrupt border segments (0-7)
        },
        colorDetails: {
          colors: ["light-brown", "dark-brown", "black", "red"], // present colors
        },
      };

      updateResults();
    }, 2000);
  }

  function updateResults() {
    if (!analysisResults) return;

    // Update scores
    asymmetryScoreEl.textContent = analysisResults.asymmetry;
    borderScoreEl.textContent = analysisResults.border;
    colorScoreEl.textContent = analysisResults.colors;
    diameterScoreEl.textContent = analysisResults.diameter.toFixed(1);

    // Calculate TDS
    const tds =
      analysisResults.asymmetry * 1.3 +
      analysisResults.border * 0.1 +
      analysisResults.colors * 0.5 +
      analysisResults.diameter * 0.5;

    tdsScoreEl.textContent = tds.toFixed(2);

    // Set interpretation
    let interpretation = "";
    let interpretationClass = "";

    if (tds < 4.75) {
      interpretation = "Benign lesion - low probability of malignancy";
      interpretationClass = "benign";
    } else if (tds >= 4.76 && tds <= 5.45) {
      interpretation = "Suspicious lesion - recommend further evaluation";
      interpretationClass = "suspicious";
    } else {
      interpretation = "Potentially malignant - urgent evaluation recommended";
      interpretationClass = "malignant";
    }

    interpretationTextEl.textContent = interpretation;
    interpretationTextEl.className = `interpretation ${interpretationClass}`;

    // ✅ Fix for Generate Report button functionality

    // Make sure the button is not disabled after segmentation
    // Inside the .then(data => { ... }) block of the handleFileSelect function

    // ENABLE BUTTON AFTER SEGMENTATION
    // Add this line after tds is computed and UI is updated:
    document.getElementById("generateReportBtn").disabled = false;

    // HANDLE REPORT GENERATION
    // Already exists at bottom of your DOMContentLoaded script:
    document
      .getElementById("generateReportBtn")
      .addEventListener("click", () => {
        fetch("/generate-report", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            asymmetry: parseFloat(
              document.getElementById("asymmetryScore").textContent
            ),
            border: parseFloat(
              document.getElementById("borderScore").textContent
            ),
            color_score: parseFloat(
              document.getElementById("colorScore").textContent
            ),
            diameter_score: parseFloat(
              document.getElementById("diameterScore").textContent
            ),
            original_filename: "latest_uploaded.png", // or dynamically update
            segmented_filename: "last_segmented.png", // match what backend saves as
            diagnosis:
              document.getElementById("interpretationText").textContent,
          }),
        })
          .then((res) => res.blob())
          .then((blob) => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "DermaScan_Report.pdf";
            document.body.appendChild(a);
            a.click();
            a.remove();
          })
          .catch((err) => {
            alert("Failed to generate report.");
            console.error(err);
          });
      });

    // Draw overlay
    drawOverlay();

    // Update chart
    updateTdsChart(tds);
  }

  function drawOverlay() {
    if (!showOverlay) {
      overlayCanvas
        .getContext("2d")
        .clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      return;
    }

    const canvas = overlayCanvas;
    const ctx = canvas.getContext("2d");

    // Set canvas dimensions to match image container
    const container = previewImage.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!analysisResults) return;

    const mode = visualizationMode.value;

    // Draw based on visualization mode
    if (mode === "combined" || mode === "asymmetry") {
      drawAsymmetry(ctx, canvas);
    }

    if (mode === "combined" || mode === "border") {
      drawBorder(ctx, canvas);
    }

    if (mode === "combined" || mode === "color") {
      drawColor(ctx, canvas);
    }
  }

  function drawAsymmetry(ctx, canvas) {
    if (!analysisResults.asymmetryDetails) return;

    const axes = analysisResults.asymmetryDetails.axes;
    const asymmetric = analysisResults.asymmetryDetails.asymmetric;

    for (let i = 0; i < axes.length; i++) {
      const axis = axes[i];
      ctx.strokeStyle = asymmetric[i]
        ? "rgba(255, 65, 108, 0.7)"
        : "rgba(76, 175, 80, 0.7)";
      ctx.lineWidth = 2;

      const x1 = axis.x1 * canvas.width;
      const y1 = axis.y1 * canvas.height;
      const x2 = axis.x2 * canvas.width;
      const y2 = axis.y2 * canvas.height;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Add axis label
      ctx.fillStyle = asymmetric[i]
        ? "rgba(255, 65, 108, 0.9)"
        : "rgba(76, 175, 80, 0.9)";
      ctx.font = "bold 12px Poppins";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2;

      // Offset the label slightly from the line
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const labelX = midX + Math.sin(angle) * 15;
      const labelY = midY - Math.cos(angle) * 15;

      ctx.fillText(asymmetric[i] ? "Asymmetric" : "Symmetric", labelX, labelY);
    }
  }

  function drawBorder(ctx, canvas) {
    if (!analysisResults.borderDetails) return;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.45;

    // Draw the lesion border
    ctx.strokeStyle = "rgba(58, 123, 213, 0.5)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Highlight abrupt segments
    const segments = analysisResults.borderDetails.segments;
    ctx.fillStyle = "rgba(58, 123, 213, 0.2)";

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      const angle1 = (segment * Math.PI) / 4 - Math.PI / 8;
      const angle2 = ((segment + 1) * Math.PI) / 4 - Math.PI / 8;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(
        centerX + radius * Math.cos(angle1),
        centerY + radius * Math.sin(angle1)
      );
      ctx.arc(centerX, centerY, radius, angle1, angle2);
      ctx.lineTo(centerX, centerY);
      ctx.fill();
    }

    // Add border label
    ctx.fillStyle = "rgba(58, 123, 213, 0.9)";
    ctx.font = "bold 12px Poppins";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(
      `${segments.length} abrupt segments`,
      centerX,
      centerY - radius - 10
    );
  }

  function drawColor(ctx, canvas) {
    if (!analysisResults.colorDetails) return;

    const colors = analysisResults.colorDetails.colors;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.3;

    // Draw color indicators
    const colorMap = {
      white: "#ffffff",
      red: "#ff416c",
      "light-brown": "#d2b48c",
      "dark-brown": "#654321",
      "blue-gray": "#7393B3",
      black: "#000000",
    };

    const angleStep = (Math.PI * 2) / colors.length;

    for (let i = 0; i < colors.length; i++) {
      const color = colors[i];
      const angle = i * angleStep;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      // Draw color dot
      ctx.fillStyle = colorMap[color] || "#cccccc";
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw connecting line
      ctx.strokeStyle = "rgba(0, 0, 0, 0.1)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();
    }

    // Add color label
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.font = "bold 12px Poppins";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(
      `${colors.length} colors detected`,
      centerX,
      centerY + radius + 15
    );
  }

  function updateTdsChart(tds) {
    const ctx = document.getElementById("tdsChart").getContext("2d");

    if (tdsChart) {
      tdsChart.destroy();
    }

    const benignThreshold = 4.75;
    const suspiciousThreshold = 5.45;

    tdsChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["TDS Score"],
        datasets: [
          {
            label: "Benign Range",
            data: [benignThreshold],
            backgroundColor: "rgba(76, 175, 80, 0.7)",
            borderColor: "rgba(76, 175, 80, 1)",
            borderWidth: 1,
          },
          {
            label: "Suspicious Range",
            data: [suspiciousThreshold - benignThreshold],
            backgroundColor: "rgba(255, 154, 68, 0.7)",
            borderColor: "rgba(255, 154, 68, 1)",
            borderWidth: 1,
          },
          {
            label: "Malignant Range",
            data: [10 - suspiciousThreshold], // Assuming max 10 for scale
            backgroundColor: "rgba(255, 65, 108, 0.7)",
            borderColor: "rgba(255, 65, 108, 1)",
            borderWidth: 1,
          },
          {
            label: "Your Score",
            data: [tds],
            backgroundColor: "rgba(58, 123, 213, 1)",
            borderColor: "rgba(58, 123, 213, 1)",
            borderWidth: 2,
            type: "line",
            pointRadius: 6,
            pointHoverRadius: 8,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            stacked: true,
            grid: {
              display: false,
            },
          },
          y: {
            stacked: true,
            beginAtZero: true,
            max: 10,
            ticks: {
              stepSize: 1,
            },
          },
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              boxWidth: 12,
              padding: 20,
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                let label = context.dataset.label || "";
                if (label) {
                  label += ": ";
                }
                if (context.datasetIndex === 3) {
                  label += context.raw.toFixed(2);
                }
                return label;
              },
            },
          },
        },
      },
    });
  }

  function generateReport() {
    // In a real implementation, this would generate a PDF report
    const reportContent = `
            <h1>DermaScan AI Analysis Report</h1>
            <h2>ABCD Rule Evaluation</h2>
            
            <div class="report-section">
                <h3>Asymmetry Score: ${analysisResults.asymmetry}</h3>
                <p>Evaluation: ${
                  analysisResults.asymmetryDetails.asymmetric.filter((a) => a)
                    .length
                } out of 2 axes showed asymmetry</p>
            </div>
            
            <div class="report-section">
                <h3>Border Score: ${analysisResults.border}</h3>
                <p>Evaluation: ${
                  analysisResults.borderDetails.segments.length
                } out of 8 segments showed abrupt cutoff</p>
            </div>
            
            <div class="report-section">
                <h3>Color Score: ${analysisResults.colors}</h3>
                <p>Colors detected: ${analysisResults.colorDetails.colors.join(
                  ", "
                )}</p>
            </div>
            
            <div class="report-section">
                <h3>Diameter: ${analysisResults.diameter.toFixed(1)} mm</h3>
            </div>
            
            <div class="report-section">
                <h2>Total Dermoscopy Score: ${(
                  analysisResults.asymmetry * 1.3 +
                  analysisResults.border * 0.1 +
                  analysisResults.colors * 0.5 +
                  analysisResults.diameter * 0.5
                ).toFixed(2)}</h2>
                <p class="interpretation">${
                  interpretationTextEl.textContent
                }</p>
            </div>
        `;

    // In a real app, you would use jsPDF or similar to generate a PDF
    // For demo purposes, we'll just show the content
    alert(
      "PDF Report Generated (simulated)\n\n" +
        reportContent.replace(/<[^>]*>/g, "")
    );
  }

  // Initialize
  drawOverlay();
  window.addEventListener("resize", drawOverlay);
});
