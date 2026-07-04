const API_URL = 'http://127.0.0.1:8000';

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const statusIndicator = document.getElementById('api-status');
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  
  const headersToggle = document.getElementById('headers-toggle');
  const headersPanel = document.getElementById('headers-panel');
  
  const extractBtn = document.getElementById('extract-btn');
  const analyzeBtn = document.getElementById('analyze-btn');
  const analyzeUrlBtn = document.getElementById('analyze-url-btn');
  const resetBtn = document.getElementById('reset-btn');
  
  const loader = document.getElementById('loader');
  const resultsArea = document.getElementById('results-area');
  const mainScanTab = document.getElementById('scan-tab');
  const mainUrlTab = document.getElementById('url-tab');
  
  // Inputs
  const emailText = document.getElementById('email-text');
  const senderName = document.getElementById('sender-name');
  const senderEmail = document.getElementById('sender-email');
  const replyToEmail = document.getElementById('reply-to-email');
  const quickUrl = document.getElementById('quick-url');
  
  // Outputs
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictDesc = document.getElementById('verdict-desc');
  
  const svmScoreVal = document.getElementById('svm-score-val');
  const svmBar = document.getElementById('svm-bar');
  const bertScoreRow = document.getElementById('bert-score-row');
  const bertScoreVal = document.getElementById('bert-score-val');
  const bertBar = document.getElementById('bert-bar');
  
  const headerResultsCard = document.getElementById('header-results-card');
  const headerStatusPill = document.getElementById('header-status-pill');
  const headerIssues = document.getElementById('header-issues');
  
  const urlResultsCard = document.getElementById('url-results-card');
  const urlAuditsList = document.getElementById('url-audits-list');
  
  // Feedback Elements
  const flagPhishBtn = document.getElementById('flag-phish-btn');
  const flagSafeBtn = document.getElementById('flag-safe-btn');
  const feedbackStatus = document.getElementById('feedback-status');

  // 1. Check API Online Status
  checkApiStatus();
  setInterval(checkApiStatus, 5000); // Check status every 5s

  async function checkApiStatus() {
    try {
      const response = await fetch(API_URL + '/');
      if (response.ok) {
        statusIndicator.innerHTML = '<span class="pulse-dot"></span> API Online';
        statusIndicator.classList.add('online');
      } else {
        throw new Error();
      }
    } catch (e) {
      statusIndicator.innerHTML = '<span class="pulse-dot"></span> API Offline';
      statusIndicator.classList.remove('online');
    }
  }

  // 2. Tab switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
      
      // Hide results if we switch tabs
      resultsArea.classList.remove('active');
    });
  });

  // 3. Headers accordion toggle
  headersToggle.addEventListener('click', () => {
    headersToggle.classList.toggle('active');
    headersPanel.classList.toggle('active');
  });

  // 4. Scan Email Button (Manual)
  analyzeBtn.addEventListener('click', () => {
    const text = emailText.value.trim();
    if (!text) {
      alert('Please enter or extract email content first.');
      return;
    }
    
    performEmailScan({
      text: text,
      sender_name: senderName.value.trim(),
      sender_email: senderEmail.value.trim(),
      reply_to: replyToEmail.value.trim()
    });
  });

  // 5. Inspect URL Button
  analyzeUrlBtn.addEventListener('click', () => {
    const url = quickUrl.value.trim();
    if (!url) {
      alert('Please enter a target URL.');
      return;
    }
    
    performUrlScan(url);
  });

  // 6. Reset view
  resetBtn.addEventListener('click', () => {
    resultsArea.classList.remove('active');
    // Clear outputs
    urlAuditsList.innerHTML = '';
    headerIssues.innerHTML = '';
    headerResultsCard.style.display = 'none';
    urlResultsCard.style.display = 'none';
    
    // Reset heights/fills
    svmBar.style.width = '0%';
    bertBar.style.width = '0%';

    // Reset feedback buttons
    flagPhishBtn.disabled = false;
    flagSafeBtn.disabled = false;
    feedbackStatus.style.display = 'none';
    feedbackStatus.innerText = '';
  });

  // 6.5. Feedback submission triggers
  flagPhishBtn.addEventListener('click', () => submitCorrection(1));
  flagSafeBtn.addEventListener('click', () => submitCorrection(0));

  async function submitCorrection(userLabel) {
    const text = emailText.value.trim() || quickUrl.value.trim();
    if (!text) {
      alert("No email text body or URL target found to scan.");
      return;
    }

    flagPhishBtn.disabled = true;
    flagSafeBtn.disabled = true;

    try {
      const payload = {
        text: text,
        sender_name: senderName.value.trim(),
        sender_email: senderEmail.value.trim(),
        reply_to: replyToEmail.value.trim(),
        user_label: userLabel
      };

      const response = await fetch(API_URL + '/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error('API server returned error status ' + response.status);
      }

      const resData = await response.json();
      feedbackStatus.innerText = userLabel === 1
        ? '✓ Logged as Phishing. Domain blacklisted. Retraining model...'
        : '✓ Logged as Safe. Retraining model...';
      feedbackStatus.style.display = 'block';
      feedbackStatus.style.color = 'var(--color-safe)';
    } catch (e) {
      alert('Error recording correction feedback: ' + e.message);
      flagPhishBtn.disabled = false;
      flagSafeBtn.disabled = false;
    }
  }

  // 7. Scrape active webmail
  extractBtn.addEventListener('click', async () => {
    showLoader();
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) {
        throw new Error('No active browser tab found.');
      }
      
      // Inject scripting to run the parser on active page
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: extractEmailContents
      }, (results) => {
        hideLoader();
        if (chrome.runtime.lastError) {
          alert('Extraction failed: Make sure you are on a Gmail or Outlook web tab, or paste the text manually.');
          console.error(chrome.runtime.lastError);
          return;
        }
        
        if (results && results[0] && results[0].result) {
          const data = results[0].result;
          
          // Pre-fill inputs
          emailText.value = data.body || '';
          senderName.value = data.senderName || '';
          senderEmail.value = data.senderEmail || '';
          replyToEmail.value = data.replyTo || '';
          
          // Auto expand headers if we have a sender email
          if (data.senderEmail) {
            headersToggle.classList.add('active');
            headersPanel.classList.add('active');
          }
          
          // Trigger scan
          performEmailScan({
            text: data.body || '',
            sender_name: data.senderName || '',
            sender_email: data.senderEmail || '',
            reply_to: data.replyTo || ''
          });
        } else {
          alert('Could not detect structure elements on this page. If you are reading an email, try selecting the email text and pasting it manually.');
        }
      });
    } catch (e) {
      hideLoader();
      alert('Error during page reading: ' + e.message);
    }
  });

  // Scanning API interactions
  async function performEmailScan(payload) {
    showLoader();
    try {
      const response = await fetch(API_URL + '/api/scan-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error('API server returned error status ' + response.status);
      }
      
      const data = await response.json();
      displayEmailResults(data);
    } catch (e) {
      alert('Scan error: Could not connect to PhishGuard API. Please run "python api.py" locally first.\n\nDetails: ' + e.message);
    } finally {
      hideLoader();
    }
  }

  async function performUrlScan(url) {
    showLoader();
    try {
      const response = await fetch(API_URL + '/api/scan-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url })
      });
      
      if (!response.ok) {
        throw new Error('API server returned error status ' + response.status);
      }
      
      const data = await response.json();
      displayUrlOnlyResults(data);
    } catch (e) {
      alert('Scan error: Could not connect to PhishGuard API. Please run "python api.py" locally first.\n\nDetails: ' + e.message);
    } finally {
      hideLoader();
    }
  }

  // Display Handlers
  function displayEmailResults(data) {
    // Reveal results area
    resultsArea.classList.add('active');
    
    // Set banner states
    verdictBanner.className = 'verdict-banner'; // reset classes
    if (data.verdict === 'Danger') {
      verdictBanner.classList.add('verdict-danger');
      verdictIcon.innerText = '🚨';
      verdictTitle.innerText = 'PHISHING THREAT DETECTED';
    } else if (data.verdict === 'Suspicious') {
      verdictBanner.classList.add('verdict-suspicious');
      verdictIcon.innerText = '⚠️';
      verdictTitle.innerText = 'SUSPICIOUS METADATA / LINK STRUCTURE';
    } else {
      verdictBanner.classList.add('verdict-safe');
      verdictIcon.innerText = '✅';
      verdictTitle.innerText = 'LEGITIMATE / SAFE STRUCTURE';
    }
    verdictDesc.innerText = data.consensus_verdict_details;
    
    // Set SVM Progress Bar
    const svmConf = data.svm.confidence * 100;
    svmScoreVal.innerText = `${svmConf.toFixed(1)}% (${data.svm.prediction === 1 ? 'Spam' : 'Safe'})`;
    svmBar.style.width = `${svmConf}%`;
    svmBar.style.background = data.svm.prediction === 1 ? 'var(--color-danger)' : 'var(--color-safe)';
    
    // Set BERT Progress Bar
    if (data.bert.available) {
      bertScoreRow.style.display = 'block';
      const bertConf = data.bert.confidence * 100;
      bertScoreVal.innerText = `${bertConf.toFixed(1)}% (${data.bert.prediction === 1 ? 'Spam' : 'Safe'})`;
      bertBar.style.width = `${bertConf}%`;
      bertBar.style.background = data.bert.prediction === 1 ? 'var(--color-danger)' : 'var(--color-safe)';
    } else {
      bertScoreRow.style.display = 'none';
    }
    
    // Render Header results
    if (data.header_audit) {
      headerResultsCard.style.display = 'block';
      headerStatusPill.className = 'status-pill';
      
      if (data.header_audit.status === 'Danger') {
        headerStatusPill.classList.add('pill-danger');
        headerStatusPill.innerText = 'Spoofed / Typosquatting';
      } else if (data.header_audit.status === 'Suspicious') {
        headerStatusPill.classList.add('pill-suspicious');
        headerStatusPill.innerText = 'Inconsistent Domain';
      } else {
        headerStatusPill.classList.add('pill-safe');
        headerStatusPill.innerText = 'Domain Integrity Passed';
      }
      
      headerIssues.innerHTML = '';
      if (data.header_audit.issues && data.header_audit.issues.length > 0) {
        data.header_audit.issues.forEach(issue => {
          const li = document.createElement('li');
          li.innerText = issue;
          headerIssues.appendChild(li);
        });
      } else {
        const li = document.createElement('li');
        li.innerText = '✓ Sender matching and Reply-To verification looks normal.';
        headerIssues.appendChild(li);
      }
    } else {
      headerResultsCard.style.display = 'none';
    }
    
    // Render URL Results
    if (data.url_audit && data.url_audit.length > 0) {
      urlResultsCard.style.display = 'block';
      urlAuditsList.innerHTML = '';
      
      data.url_audit.forEach(audit => {
        const itemCard = document.createElement('div');
        itemCard.className = 'link-audit-card';
        
        const badgeClass = audit.status === 'Danger' ? 'badge-danger' : audit.status === 'Suspicious' ? 'badge-suspicious' : 'badge-safe';
        const label = audit.status === 'Danger' ? 'Danger' : audit.status === 'Suspicious' ? 'Suspicious' : 'Safe Structure';
        
        let issuesListHtml = '';
        if (audit.issues && audit.issues.length > 0) {
          issuesListHtml = `<ul class="issues-list" style="margin-top:8px;">${audit.issues.map(i => `<li>${i}</li>`).join('')}</ul>`;
        } else {
          issuesListHtml = `<p class="helper-text" style="color:var(--color-safe); margin-top:4px;">✓ Domain and lexical verification passed.</p>`;
        }
        
        itemCard.innerHTML = `
          <span class="link-url">${audit.url}</span>
          <span class="link-badge ${badgeClass}">${label}</span>
          ${issuesListHtml}
        `;
        urlAuditsList.appendChild(itemCard);
      });
    } else {
      urlResultsCard.style.display = 'none';
    }
  }

  function displayUrlOnlyResults(audit) {
    // Reveal results area
    resultsArea.classList.add('active');
    
    // Set banner states
    verdictBanner.className = 'verdict-banner'; // reset classes
    if (audit.status === 'Danger') {
      verdictBanner.classList.add('verdict-danger');
      verdictIcon.innerText = '🚨';
      verdictTitle.innerText = 'MALICIOUS LINK DETECTED';
    } else if (audit.status === 'Suspicious') {
      verdictBanner.classList.add('verdict-suspicious');
      verdictIcon.innerText = '⚠️';
      verdictTitle.innerText = 'SUSPICIOUS HEURISTICS';
    } else {
      verdictBanner.classList.add('verdict-safe');
      verdictIcon.innerText = '✅';
      verdictTitle.innerText = 'SAFE LINK STRUCTURE';
    }
    verdictDesc.innerText = audit.issues.length > 0 ? 'Heuristic/Classifier triggers detected' : 'Domain structure looks official and safe';
    
    // Hide text model scores
    svmScoreVal.innerText = 'N/A';
    svmBar.style.width = '0%';
    bertScoreRow.style.display = 'none';
    
    // Hide Header results
    headerResultsCard.style.display = 'none';
    
    // Show URL Audit results
    urlResultsCard.style.display = 'block';
    urlAuditsList.innerHTML = '';
    
    const itemCard = document.createElement('div');
    itemCard.className = 'link-audit-card';
    
    const badgeClass = audit.status === 'Danger' ? 'badge-danger' : audit.status === 'Suspicious' ? 'badge-suspicious' : 'badge-safe';
    const label = audit.status === 'Danger' ? 'Danger' : audit.status === 'Suspicious' ? 'Suspicious' : 'Safe';
    
    let issuesListHtml = '';
    if (audit.issues && audit.issues.length > 0) {
      issuesListHtml = `<ul class="issues-list" style="margin-top:8px;">${audit.issues.map(i => `<li>${i}</li>`).join('')}</ul>`;
    } else {
      issuesListHtml = `<p class="helper-text" style="color:var(--color-safe); margin-top:4px;">✓ Link structure matches typical whitelisted patterns.</p>`;
    }
    
    itemCard.innerHTML = `
      <span class="link-url">${audit.url}</span>
      <span class="link-badge ${badgeClass}">${label}</span>
      ${issuesListHtml}
    `;
    urlAuditsList.appendChild(itemCard);
  }

  function showLoader() {
    loader.classList.add('active');
  }

  function hideLoader() {
    loader.classList.remove('active');
  }
});

// THIS FUNCTION IS INJECTED AS A SCRIPT AND RUNS IN THE ACTIVE PAGE CONTEXT
function extractEmailContents() {
  const result = {
    body: '',
    senderName: '',
    senderEmail: '',
    replyTo: ''
  };

  const hostname = window.location.hostname;
  
  if (hostname.includes('mail.google.com')) {
    // Gmail extractors
    
    // 1. Get Sender Info
    // Gmail sender display names and email addresses are usually stored in attributes of elements with class 'gD' or inside the detailed header area
    const senderElement = document.querySelector('.gD');
    if (senderElement) {
      result.senderName = senderElement.getAttribute('name') || '';
      result.senderEmail = senderElement.getAttribute('email') || '';
    }
    
    // 2. Get Reply-To Info (Class-independent double-failsafe)
    // Failsafe A: Scan all table rows in the document for key-value labels
    const allRows = document.querySelectorAll('tr');
    allRows.forEach(row => {
      const text = row.textContent.toLowerCase();
      if (text.includes('reply-to:') || text.includes('reply-to')) {
        const cells = row.querySelectorAll('td');
        if (cells.length > 1) {
          if (cells[0].textContent.toLowerCase().includes('reply-to') || cells[0].textContent.toLowerCase().includes('reply_to')) {
            result.replyTo = cells[1].textContent.trim();
          }
        }
      }
    });

    // Failsafe B: If Failsafe A missed it, scan for any label element containing 'reply-to:' text
    if (!result.replyTo) {
      const allLabels = document.querySelectorAll('td, span, div, b');
      for (const el of allLabels) {
        const txt = el.textContent.trim().toLowerCase();
        if (txt === 'reply-to:' || txt === 'reply-to') {
          const parent = el.parentElement;
          if (parent) {
            const siblings = parent.querySelectorAll('td, span, div');
            if (siblings.length > 1) {
              result.replyTo = siblings[siblings.length - 1].textContent.trim();
              break;
            }
          }
        }
      }
    }
    
    // 3. Get Body Text
    // Gmail stores email content in elements with role='gridcell' or class 'a3s'
    const bodyElements = document.querySelectorAll('.a3s');
    if (bodyElements && bodyElements.length > 0) {
      // Grab the last email body (in case of threads)
      const activeBody = bodyElements[bodyElements.length - 1];
      result.body = activeBody.innerText || activeBody.textContent || '';
    }
    
  } else if (hostname.includes('outlook.live.com') || hostname.includes('outlook.office.com') || hostname.includes('outlook.office365.com')) {
    // Outlook Web extractors
    
    // 1. Sender name & email
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
    
    // 2. Body Text
    // Outlook uses classes like 'x_MsoNormal' or div[role="document"] inside message panes
    const bodyElement = document.querySelector('[role="document"], .x_MsoNormal, .ReadingPaneBody');
    if (bodyElement) {
      result.body = bodyElement.innerText || bodyElement.textContent || '';
    }
  }
  
  // Generic Fallback if nothing specific matched
  if (!result.body) {
    // Try to get selected text from user's current selection
    const selection = window.getSelection().toString().trim();
    if (selection) {
      result.body = selection;
    } else {
      // Scrape general central page content
      const article = document.querySelector('article') || document.querySelector('main');
      if (article) {
        result.body = article.innerText.slice(0, 5000);
      } else {
        result.body = document.body.innerText.slice(0, 5000);
      }
    }
  }

  // Clean raw email addresses (extract values inside brackets <...>)
  const cleanEmail = (emailStr) => {
    if (!emailStr) return '';
    const match = emailStr.match(/<([^>]+)>/);
    return match ? match[1].trim() : emailStr.trim();
  };

  result.senderEmail = cleanEmail(result.senderEmail);
  result.replyTo = cleanEmail(result.replyTo);
  
  return result;
}
