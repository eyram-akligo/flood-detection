\# Flood Detection Using Transfer Learning on DenseNet121

This project uses transfer learning on the DenseNet121 architecture to
automate the detection of flood events from satellite images. The model
was trained on a custom dataset of flood images and tested on real-world
satellite data.

\## Features - \*\*Transfer Learning\*\*: Built on the DenseNet121 deep
learning model for high accuracy. - \*\*Interactive Web App\*\*: A
user-friendly interface developed with Streamlit. - \*\*Input
Format\*\*: Upload \`.tif\` satellite images for flood detection.

\## Getting Started

\### Prerequisites - Python 3.8 or later - Streamlit - TensorFlow - GDAL
(for \`.tif\` image processing)

\### Installation 1. Clone this repository: \`\`\`bash git clone
https://github.com/yourusername/flood-detection.git cd flood-detection
\`\`\`

2\. Install dependencies: \`\`\`bash pip install -r app/requirements.txt
\`\`\`

3\. Run the Streamlit app: \`\`\`bash streamlit run app/streamlit_app.py
\`\`\`

4\. Open the app in your browser at \`http://localhost:8501\`.

\### Usage 1. Launch the Streamlit app. 2. Upload a \`.tif\` image via
the interface. 3. The app will display flood detection results.

\### Example Upload the provided example file in the \`example_data/\`
folder for testing the app.

\## Model Details - \*\*Architecture\*\*: DenseNet121 -
\*\*Training\*\*: Transfer learning with pre-trained weights -
\*\*Dataset\*\*: A curated dataset of flood and non-flood satellite
images.

\## Acknowledgments - Pre-trained DenseNet121 weights from
(https://keras.io/api/applications/densenet/). - Satellite imagery
courtesy of Sentinel-2 (Sen12) on Kaggle. - Source code from
KonstantinosF https://github.com/KonstantinosF
