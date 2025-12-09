// Multi-language support with JSON files
let currentTranslations = {};

async function loadTranslations(lang) {
    try {
        const response = await fetch(`/static/lang/${lang}.json`);
        if (!response.ok) throw new Error('Translation file not found');
        return await response.json();
    } catch (error) {
        console.error('Error loading translations:', error);
        // Fallback to English
        if (lang !== 'en') {
            return await loadTranslations('en');
        }
        return {};
    }
}

function getNestedValue(obj, path) {
    return path.split('.').reduce((current, key) => current?.[key], obj);
}

async function changeLanguage(lang) {
    localStorage.setItem('selectedLanguage', lang);
    currentTranslations = await loadTranslations(lang);
    
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = getNestedValue(currentTranslations, key);
        if (translation) {
            element.textContent = translation;
        }
    });

    // Update placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        const translation = getNestedValue(currentTranslations, key);
        if (translation) {
            element.placeholder = translation;
        }
    });

    // Update titles
    document.querySelectorAll('[data-i18n-title]').forEach(element => {
        const key = element.getAttribute('data-i18n-title');
        const translation = getNestedValue(currentTranslations, key);
        if (translation) {
            element.title = translation;
        }
    });

    // Update innerHTML for elements with HTML content
    document.querySelectorAll('[data-i18n-html]').forEach(element => {
        const key = element.getAttribute('data-i18n-html');
        const translation = getNestedValue(currentTranslations, key);
        if (translation) {
            element.innerHTML = translation;
        }
    });
}

// Load saved language on page load
window.addEventListener('DOMContentLoaded', async function() {
    const savedLang = localStorage.getItem('selectedLanguage') || 'en';
    const langSelect = document.getElementById('languageSelect');
    if (langSelect) {
        langSelect.value = savedLang;
    }
    await changeLanguage(savedLang);
});

// Export for use in other scripts
window.i18n = {
    changeLanguage,
    getCurrentTranslations: () => currentTranslations,
    t: (key) => getNestedValue(currentTranslations, key) || key
};
