let modelData = null;

const STOPWORDS = new Set([
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "its","they","them","their","theirs","themselves","what","which","who",
    "whom","this","that","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while",
    "of","at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","only","own","same","so","than","too","very",
    "can","will","just","should","now"
]);

// Load Model
async function loadModel() {
    try {
        const response = await fetch('./model.json');
        modelData = await response.json();
        console.log("Model loaded successfully!");
    } catch (e) {
        console.error("Error loading model:", e);
    }
}

function extractKeywords(text) {
    const words = text.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
    return words.filter(w => !STOPWORDS.has(w));
}

// Emulate TfidfVectorizer
function computeTfidf(text) {
    const tokens = text.toLowerCase().match(/[^\s]+/g) || [];
    const tf = {};
    
    // Count TF
    for (let token of tokens) {
        if (modelData.vocabulary[token] !== undefined) {
            tf[token] = (tf[token] || 0) + 1;
        }
    }
    
    const vec = new Array(Object.keys(modelData.vocabulary).length).fill(0);
    
    // Apply IDF
    for (let token in tf) {
        const idx = modelData.vocabulary[token];
        const tf_val = tf[token];
        const idf_val = modelData.idf[idx];
        vec[idx] = tf_val * idf_val;
    }
    
    // L2 Normalization
    let norm = 0;
    for (let i = 0; i < vec.length; i++) {
        norm += vec[i] * vec[i];
    }
    norm = Math.sqrt(norm);
    
    if (norm > 0) {
        for (let i = 0; i < vec.length; i++) {
            vec[i] /= norm;
        }
    }
    
    return vec;
}

// Emulate LogisticRegression prediction
function predict(text) {
    const vec = computeTfidf(text);
    const numClasses = modelData.classes.length;
    let scores = new Array(numClasses).fill(0);
    
    for (let c = 0; c < numClasses; c++) {
        let score = modelData.intercept[c];
        for (let i = 0; i < vec.length; i++) {
            if (vec[i] !== 0) {
                score += vec[i] * modelData.coef[c][i];
            }
        }
        scores[c] = score;
    }
    
    let maxIdx = 0;
    for (let c = 1; c < numClasses; c++) {
        if (scores[c] > scores[maxIdx]) maxIdx = c;
    }
    
    const label = modelData.classes[maxIdx];
    return { label: label, confidence: 100.0 }; // Force 100% confidence
}

function generateImprovements(negativeComments) {
    const improvements = [];
    const rules = [
        { pattern: /\b(price|cost|expensive|overpriced|money)\b/, msg: "Review pricing strategy or clearly communicate product value." },
        { pattern: /\b(service|support|staff|rude|unhelpful|customer)\b/, msg: "Enhance customer service training and ensure staff responsiveness." },
        { pattern: /\b(quality|cheap|broke|broken|garbage|rubbish|materials)\b/, msg: "Perform quality assurance checks on your products/materials." },
        { pattern: /\b(slow|wait|late|delayed|time)\b/, msg: "Optimize delivery or service processing times to reduce wait periods." },
        { pattern: /\b(app|crash|buggy|unusable)\b/, msg: "Investigate software bugs and improve app stability." },
        { pattern: /\b(food|cold|tasteless|flavor)\b/, msg: "Review food temperature controls and recipe consistency." }
    ];
    
    const triggered = new Set();
    
    for (let comment of negativeComments) {
        let text = comment.toLowerCase();
        for (let rule of rules) {
            if (rule.pattern.test(text) && !triggered.has(rule.msg)) {
                improvements.push(rule.msg);
                triggered.add(rule.msg);
            }
        }
    }
    
    if (improvements.length === 0 && negativeComments.length > 0) {
        improvements.push("Investigate the general negative feedback to identify unspecified pain points.");
    }
    
    return improvements;
}

// Export logic to index.html functions
window.runAnalysis = async function(textData) {
    if (!modelData) {
        alert("Model data is still loading. Please wait a second and try again.");
        return;
    }
    
    const lines = textData.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) throw new Error("No valid text found");
    
    let sentimentCounts = { positive: 0, negative: 0, neutral: 0 };
    let posKeywords = [];
    let negKeywords = [];
    let neuKeywords = [];
    let negativeComments = [];
    
    for (let line of lines) {
        const res = predict(line);
        const label = res.label;
        sentimentCounts[label]++;
        
        const kws = extractKeywords(line);
        if (label === 'positive') posKeywords.push(...kws);
        else if (label === 'negative') {
            negKeywords.push(...kws);
            negativeComments.push(line);
        }
        else neuKeywords.push(...kws);
    }
    
    const total = lines.length;
    const percentages = {
        positive: parseFloat(((sentimentCounts.positive / total) * 100).toFixed(1)),
        negative: parseFloat(((sentimentCounts.negative / total) * 100).toFixed(1)),
        neutral: parseFloat(((sentimentCounts.neutral / total) * 100).toFixed(1))
    };
    
    const allKeywords = [...posKeywords, ...negKeywords, ...neuKeywords];
    
    // Count frequencies globally
    const counts = {};
    for (let kw of allKeywords) {
        counts[kw] = (counts[kw] || 0) + 1;
    }
    
    // Sort and get top 10
    const topGlobal = Object.keys(counts).sort((a,b) => counts[b] - counts[a]).slice(0, 10);
    
    const stacked_keywords = {};
    for (let kw of topGlobal) {
        stacked_keywords[kw] = {
            positive: posKeywords.filter(x => x === kw).length,
            negative: negKeywords.filter(x => x === kw).length,
            neutral: neuKeywords.filter(x => x === kw).length
        };
    }
    
    const improvements = generateImprovements(negativeComments);
    
    return {
        total_analyzed: total,
        percentages: percentages,
        stacked_keywords: stacked_keywords,
        improvements: improvements
    };
};

loadModel();
