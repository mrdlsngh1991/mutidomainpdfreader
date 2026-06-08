import { Routes, Route } from "react-router-dom";
//import Home from "./pages/Home";
import ChatBot from "./Pages/chatbot";
import "./App.css";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ChatBot />} />
    </Routes>
  );
}