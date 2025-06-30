# Sound Analyzer

Sound Analyzer is a web-based application that listens to audio input, analyzes the sound frequencies in real time, and visualizes them by highlighting corresponding keys on a virtual piano keyboard. This tool is designed for musicians, educators, students, and anyone interested in understanding or visualizing the frequencies present in musical or environmental sounds.

---

## Features

- **Real-time Sound Analysis:** Captures audio from your microphone and detects its frequency spectrum.
- **Piano Visualization:** Maps detected frequencies to notes and highlights the matching keys on a virtual piano.
- **Multi-language Codebase:** Built with Python (backend), TypeScript/JavaScript (frontend), and HTML/CSS for UI.
- **Educational Tool:** Great for music theory education, ear training, and acoustics demonstrations.
- **User-friendly Interface:** Simple, interactive design accessible from any modern web browser.

---

## Demo

![Demo Screenshot](https://i.imgur.com/ZGmdxSW.png)  
*Example: The app listening to a note and lighting up the corresponding piano key.*

---

## Getting Started

### Prerequisites

- Python 3.x
- Node.js and npm (for frontend assets)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/mbielicki/sound-analyzer.git
    cd sound-analyzer
    ```

2. **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure the base URL in frontend/src/config.js**

4. **Install frontend dependencies:**
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

5. **Run the application:**
    ```bash
    python manage.py runserver
    ```

6. **Open in your browser:**  
    Visit [http://localhost:8000](http://localhost:8000) (or the indicated URL) to start analyzing sound.

---

## Usage

- **Allow access to your microphone** when prompted.
- Play or sing a note, or play an instrument.
- Watch as the app analyzes the audio and highlights the keys on the virtual piano that correspond to the detected frequencies.

---

## Technologies Used

- **Python:** Backend audio processing and server logic.
- **TypeScript / JavaScript:** Real-time frontend visualization and audio input handling.
- **HTML / CSS:** Responsive UI and virtual piano rendering.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add Your Feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Inspired by digital pianos and music visualization tools.
- Special thanks to the open-source community for supporting audio analysis libraries.

---

## Contact

Created by [mbielicki](https://github.com/mbielicki)  
For questions or suggestions, please open an issue on GitHub.

---

*Enjoy visualizing and understanding sound!*
