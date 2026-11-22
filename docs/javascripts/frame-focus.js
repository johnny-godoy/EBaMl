// Chrome gives a frame that holds the focus no :focus state, so CSS alone cannot draw a focus ring on it. A reader who
// moves into an embedded notebook or paper with the Tab key then cannot see where the focus is. This script marks the
// frame that holds the focus with the class `ebaml-focused`, and extra.css draws the ring.
(() => {
  const mark = () => {
    for (const frame of document.querySelectorAll("iframe.ebaml-focused")) {
      frame.classList.remove("ebaml-focused");
    }
    if (document.activeElement instanceof HTMLIFrameElement) {
      document.activeElement.classList.add("ebaml-focused");
    }
  };
  // The page loses the focus to the frame, and document.activeElement changes after the blur event.
  window.addEventListener("blur", () => setTimeout(mark));
  window.addEventListener("focus", mark);
})();
