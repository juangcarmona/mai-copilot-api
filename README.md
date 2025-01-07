# MAI Copilot API

The **MAI Copilot API** is a local-first AI-powered API for code assistance. Built with **FastAPI**, it's designed to be fast, lightweight, and extensible.

## Features
- **Local-first:** Designed to run on your machine.
- **AI-powered:** Provides intelligent coding suggestions.
- **Extensible:** Easy to integrate and customize.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/juangcarmona/mai-copilot-api.git
cd mai-copilot-api
```

### 2. Install Dependencies
Ensure you have Python 3.10 or later installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the API
Start the server using Uvicorn:
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

## Endpoints

### `POST /api/generate/`
Generates code suggestions based on the provided input prompt.

#### Request Body:
```json
{
  "inputs": "<prompt>",
  "parameters": {
    "temperature": 0.7,
    "max_length": 100
  }
}
```

#### Response:
```json
{
  "generated_text": "<generated_code>",
  "status": 200
}
```

### `GET /`
Redirects to the API documentation (Swagger UI).

---

## Configuration for VSCode
To use this API with a compatible VSCode extension, set the following in your `settings.json` file:
```json
{
  "llm.url": "http://localhost:8000/api/generate/",
  "llm.disableUrlPathCompletion": true,
  "llm.tlsSkipVerifyInsecure": true,
  "llm.fillInTheMiddle.enabled": true
}
```

---

## Contributing

Contributions are welcome! Feel free to:
- Open issues for bug reports or feature requests.
- Submit pull requests with improvements or fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
