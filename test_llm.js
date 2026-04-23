import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { generateOpeningMessage } from "./backend/src/services/llm.provider.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, "backend/.env") });

async function run() {
  const submission = {
    title: "Test Image",
    inputType: "image",
    uploadedFileUrl: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP...",
    uploadedFileType: "image/jpeg"
  };

  try {
    const res = await generateOpeningMessage(submission);
    console.log("Result:", res);
  } catch (err) {
    console.error("Error:", err);
  }
}

run();
