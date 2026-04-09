import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

type ErrorBoundaryState = {
  hasError: boolean;
  message: string;
};

class ErrorBoundary extends React.Component<
  React.PropsWithChildren,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = {
    hasError: false,
    message: ""
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      message: error.message
    };
  }

  override componentDidCatch(error: Error): void {
    // Surface error details in devtools while keeping UI visible.
    // eslint-disable-next-line no-console
    console.error("Track Builder UI runtime error:", error);
  }

  override render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: "20px",
            color: "#e7ecf6",
            background: "#101522",
            fontFamily: "Segoe UI, sans-serif"
          }}
        >
          <h2 style={{ marginTop: 0 }}>UI Runtime Error</h2>
          <p>{this.state.message || "Unknown error"}</p>
          <p>Open browser console for full stack trace.</p>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
