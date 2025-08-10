require('dotenv').config(); // add this at the top of your main file

const hfToken = process.env.HUGGINGFACE_TOKEN;
const apiKey = process.env.API_KEY;

console.log("Hugging Face Token:", hfToken);
console.log("Google API Key:", apiKey);
