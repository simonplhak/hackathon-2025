document.addEventListener("DOMContentLoaded", () => {
    // --- App State ---
    const state = {
        history: [],
        currentDisplay: "0",
        isError: false,
    };

    // --- DOM Elements ---
    const display = document.getElementById("display");
    const navButtons = document.querySelectorAll(".nav-btn");
    const contentSections = document.querySelectorAll(".content-section");
    const calculatorGrid = document.getElementById("calculator-grid");
    const historyList = document.getElementById("history-list");
    
    // --- Unit Converter Elements ---
    const unitCategorySelect = document.getElementById("unit-category");
    const fromUnitSelect = document.getElementById("from-unit");
    const toUnitSelect = document.getElementById("to-unit");
    const unitInput = document.getElementById("unit-input");
    const unitResult = document.getElementById("unit-result");
    const convertUnitBtn = document.getElementById("convert-unit-btn");

    // --- Currency Converter Elements ---
    const fromCurrencySelect = document.getElementById("from-currency");
    const toCurrencySelect = document.getElementById("to-currency");
    const currencyInput = document.getElementById("currency-input");
    const currencyResult = document.getElementById("currency-result");
    const convertCurrencyBtn = document.getElementById("convert-currency-btn");

    // --- Data ---
    const units = {
        length: { "Meters": 1, "Kilometers": 1000, "Miles": 1609.34, "Feet": 0.3048 },
        weight: { "Grams": 1, "Kilograms": 1000, "Pounds": 453.592 },
        temperature: { "Celsius": "c", "Fahrenheit": "f", "Kelvin": "k" }
    };

    // --- Navigation ---
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.id.replace("nav-", "") + "-section";
            navButtons.forEach(b => b.classList.remove("active"));
            contentSections.forEach(s => s.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // --- Calculator Logic ---
    function updateDisplay() {
        display.value = state.currentDisplay;
    }

    function handleCalculatorInput(value) {
        if (state.isError) {
            state.currentDisplay = "0";
            state.isError = false;
        }

        if (value >= "0" && value <= "9") {
            if (state.currentDisplay === "0") {
                state.currentDisplay = value;
            } else {
                state.currentDisplay += value;
            }
        } else if (value === ".") {
            if (!state.currentDisplay.includes(".")) {
                state.currentDisplay += ".";
            }
        } else if (["+", "-", "*", "/"].includes(value)) {
            if (state.currentDisplay !== "0" && !["+", "-", "*", "/"].includes(state.currentDisplay.slice(-1))) {
                state.currentDisplay += value;
            }
        } else if (value === "C") {
            state.currentDisplay = "0";
        } else if (value === "del") {
            state.currentDisplay = state.currentDisplay.slice(0, -1) || "0";
        } else if (value === "=") {
            calculate();
        } else {
            handleFunction(value);
        }
        updateDisplay();
    }
    
    function handleFunction(func) {
        try {
            const current = parseFloat(state.currentDisplay);
            if(isNaN(current)) throw new Error();
            let result;
            switch(func) {
                case "sqrt": result = Math.sqrt(current); break;
                case "pow": result = Math.pow(current, 2); break;
                case "%": result = current / 100; break;
                case "sin": result = Math.sin(current * Math.PI / 180); break; // Degrees to Radians
                case "cos": result = Math.cos(current * Math.PI / 180); break;
                case "tan": result = Math.tan(current * Math.PI / 180); break;
                default: return;
            }
            if (!isFinite(result)) throw new Error("Invalid Result");
            const formattedResult = parseFloat(result.toPrecision(10));
            addToHistory(`${func}(${state.currentDisplay}) = ${formattedResult}`);
            state.currentDisplay = formattedResult.toString();
        } catch (e) {
            state.currentDisplay = "Error";
            state.isError = true;
        }
    }

    function calculate() {
        try {
            if (/\/0/.test(state.currentDisplay)) {
                throw new Error("Division by zero");
            }
            // Using Function constructor instead of eval for slightly better security in this context.
            const result = new Function("return " + state.currentDisplay)();
            if (!isFinite(result)) throw new Error("Invalid Result");
            
            const formattedResult = parseFloat(result.toPrecision(10));
            addToHistory(`${state.currentDisplay} = ${formattedResult}`);
            state.currentDisplay = formattedResult.toString();
        } catch (error) {
            state.currentDisplay = "Error";
            state.isError = true;
        }
    }

    calculatorGrid.addEventListener("click", (e) => {
        if (e.target.tagName === "BUTTON") {
            handleCalculatorInput(e.target.dataset.value);
        }
    });

    // --- History Logic ---
    function addToHistory(entry) {
        state.history.unshift(entry);
        if (state.history.length > 20) {
            state.history.pop();
        }
        renderHistory();
    }

    function renderHistory() {
        if (state.history.length === 0) {
            historyList.innerHTML = "<li>No calculations yet.</li>";
        } else {
            historyList.innerHTML = state.history.map(item => `<li>${item}</li>`).join("");
        }
    }

    // --- Unit Converter Logic ---
    function populateUnitSelectors() {
        const category = unitCategorySelect.value;
        const options = Object.keys(units[category]);
        fromUnitSelect.innerHTML = options.map(o => `<option value="${o}">${o}</option>`).join("");
        toUnitSelect.innerHTML = options.map(o => `<option value="${o}">${o}</option>`).join("");
    }

    function convertUnits() {
        const category = unitCategorySelect.value;
        const from = fromUnitSelect.value;
        const to = toUnitSelect.value;
        const input = parseFloat(unitInput.value);

        if (isNaN(input)) {
            unitResult.value = "Invalid Input";
            return;
        }

        let result;
        if (category === "temperature") {
            // Temperature conversion requires specific formulas
            if (from === to) {
                result = input;
            } else if (from === "Celsius" && to === "Fahrenheit") {
                result = (input * 9/5) + 32;
            } else if (from === "Fahrenheit" && to === "Celsius") {
                result = (input - 32) * 5/9;
            } else if (from === "Celsius" && to === "Kelvin") {
                result = input + 273.15;
            } else if (from === "Kelvin" && to === "Celsius") {
                result = input - 273.15;
            } else if (from === "Fahrenheit" && to === "Kelvin") {
                result = (input - 32) * 5/9 + 273.15;
            } else if (from === "Kelvin" && to === "Fahrenheit") {
                result = (input - 273.15) * 9/5 + 32;
            }
        } else {
            // Standard unit conversion based on a base unit (Meters/Grams)
            const fromValue = units[category][from];
            const toValue = units[category][to];
            const baseValue = input * fromValue;
            result = baseValue / toValue;
        }
        unitResult.value = parseFloat(result.toFixed(5));
    }

    unitCategorySelect.addEventListener("change", populateUnitSelectors);
    convertUnitBtn.addEventListener("click", convertUnits);

    // --- Currency Converter Logic ---
    // MOCK API: In a real app, replace this with a fetch call to a real currency API.
    const mockRates = {
        "USD": 1, "EUR": 0.92, "GBP": 0.79, "JPY": 157.25, "CAD": 1.37, "AUD": 1.50
    };

    function populateCurrencySelectors() {
        const currencies = Object.keys(mockRates);
        const options = currencies.map(c => `<option value="${c}">${c}</option>`).join("");
        fromCurrencySelect.innerHTML = options;
        toCurrencySelect.innerHTML = options;
        fromCurrencySelect.value = "USD";
        toCurrencySelect.value = "EUR";
    }

    async function convertCurrency() {
        const from = fromCurrencySelect.value;
        const to = toCurrencySelect.value;
        const amount = parseFloat(currencyInput.value);

        if (isNaN(amount)) {
            currencyResult.value = "Invalid Amount";
            return;
        }

        try {
            // This simulates an API call. Replace with a real `fetch` in production.
            // const response = await fetch(`https://api.exchangerate-api.com/v4/latest/${from}`);
            // if (!response.ok) throw new Error("Network response was not ok");
            // const data = await response.json();
            // const rate = data.rates[to];
            
            // Using mock data for demonstration
            const fromRate = mockRates[from];
            const toRate = mockRates[to];
            if (!fromRate || !toRate) throw new Error("Currency not found");
            
            const result = (amount / fromRate) * toRate;
            currencyResult.value = `${result.toFixed(2)} ${to}`;

        } catch (error) {
            currencyResult.value = "Error fetching rate";
            console.error("Currency conversion error:", error);
        }
    }

    convertCurrencyBtn.addEventListener("click", convertCurrency);

    // --- Initial Setup ---
    function initialize() {
        updateDisplay();
        renderHistory();
        populateUnitSelectors();
        populateCurrencySelectors();
    }

    initialize();
});