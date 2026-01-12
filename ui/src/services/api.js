import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Ask a question to the TiddlyWiki AI backend
 * @param {string} question - The question to ask
 * @param {number} topK - Number of relevant tiddlers to retrieve (default: 5)
 * @param {string} model - OpenAI model to use (default: gpt-4o-mini)
 * @returns {Promise<{question: string, answer: string, sources: Array}>}
 */
export async function askQuestion(question, topK = 5, model = 'gpt-4o-mini') {
  try {
    const response = await axios.post(`${API_BASE_URL}/ask`, {
      question,
      top_k: topK,
      model,
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.error || 'Server error occurred');
    } else if (error.request) {
      // Request made but no response received
      throw new Error('Unable to connect to the server. Please ensure the backend is running.');
    } else {
      // Something else happened
      throw new Error('An unexpected error occurred');
    }
  }
}
