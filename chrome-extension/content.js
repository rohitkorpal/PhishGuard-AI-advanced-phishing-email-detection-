// PhishGuard AI - Content Script
// This script runs in the context of webmail pages (Gmail, Outlook).

console.log("🛡️ PhishGuard AI content protection active on this tab.");

// Listener for messages from the popup or background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "ping") {
    sendResponse({ status: "active" });
    return true;
  }
  
  if (request.action === "extract_current_email") {
    // Re-use parser logic
    try {
      const emailDetails = extractPageDetails();
      sendResponse({ success: true, data: emailDetails });
    } catch (e) {
      sendResponse({ success: false, error: e.message });
    }
    return true;
  }
});

function extractPageDetails() {
  const result = {
    body: '',
    senderName: '',
    senderEmail: '',
    replyTo: ''
  };

  const hostname = window.location.hostname;
  
  if (hostname.includes('mail.google.com')) {
    // Gmail DOM scrapers
    const senderElement = document.querySelector('.gD');
    if (senderElement) {
      result.senderName = senderElement.getAttribute('name') || '';
      result.senderEmail = senderElement.getAttribute('email') || '';
    }
    
    const bodyElements = document.querySelectorAll('.a3s');
    if (bodyElements && bodyElements.length > 0) {
      const activeBody = bodyElements[bodyElements.length - 1];
      result.body = activeBody.innerText || activeBody.textContent || '';
    }
  } else if (hostname.includes('outlook')) {
    // Outlook DOM scrapers
    const senderNode = document.querySelector('[data-type="sender"] button, [data-testid="PersonaSenderName"]');
    if (senderNode) {
      result.senderName = senderNode.textContent.trim();
    }
    
    const senderEmailNode = document.querySelector('[data-testid="PersonaSenderEmail"], [title*="@"]');
    if (senderEmailNode) {
      const title = senderEmailNode.getAttribute('title') || '';
      if (title.includes('@')) {
        result.senderEmail = title;
      }
    }
    
    const bodyElement = document.querySelector('[role="document"], .x_MsoNormal, .ReadingPaneBody');
    if (bodyElement) {
      result.body = bodyElement.innerText || bodyElement.textContent || '';
    }
  }
  
  // Selection fallback
  if (!result.body) {
    result.body = window.getSelection().toString().trim();
  }
  
  return result;
}
