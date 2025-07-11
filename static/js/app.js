// Influencer Discovery Tool - Frontend JavaScript

class InfluencerDiscoveryApp {
    constructor() {
        this.currentSearchQuery = '';
        this.suggestionTimeout = null;
        this.init();
    }

    init() {
        this.loadSystemStats();
        this.loadCategories();
        this.setupEventListeners();
        this.setupSearchSuggestions();
    }

    // Load system statistics
    async loadSystemStats() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                document.getElementById('total-influencers').textContent = data.total_influencers;
            }
        } catch (error) {
            console.error('Failed to load system stats:', error);
            document.getElementById('total-influencers').textContent = 'Error';
        }
    }

    // Load available categories
    async loadCategories() {
        try {
            const response = await fetch('/api/categories');
            const data = await response.json();
            
            const categorySelect = document.getElementById('categoryFilter');
            data.categories.forEach(category => {
                const option = document.createElement('option');
                option.value = category;
                option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
                categorySelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load categories:', error);
        }
    }

    // Setup event listeners
    setupEventListeners() {
        // Search form submission
        document.getElementById('searchForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.performSearch();
        });

        // Search input for suggestions
        const searchInput = document.getElementById('searchQuery');
        searchInput.addEventListener('input', (e) => {
            this.handleSearchInput(e.target.value);
        });

        // Hide suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#searchQuery') && !e.target.closest('#suggestions')) {
                this.hideSuggestions();
            }
        });

        // Keyboard navigation for suggestions
        searchInput.addEventListener('keydown', (e) => {
            this.handleSuggestionNavigation(e);
        });
    }

    // Setup search suggestions
    setupSearchSuggestions() {
        const suggestionsContainer = document.getElementById('suggestions');
        
        // Handle suggestion clicks
        suggestionsContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-item')) {
                const query = e.target.textContent;
                document.getElementById('searchQuery').value = query;
                this.hideSuggestions();
                this.performSearch();
            }
        });
    }

    // Handle search input for suggestions
    handleSearchInput(query) {
        clearTimeout(this.suggestionTimeout);
        
        if (query.length < 2) {
            this.hideSuggestions();
            return;
        }

        this.suggestionTimeout = setTimeout(() => {
            this.loadSuggestions(query);
        }, 300);
    }

    // Load search suggestions
    async loadSuggestions(query) {
        try {
            const response = await fetch(`/api/suggestions?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            this.displaySuggestions(data.suggestions);
        } catch (error) {
            console.error('Failed to load suggestions:', error);
        }
    }

    // Display suggestions dropdown
    displaySuggestions(suggestions) {
        const container = document.getElementById('suggestions');
        
        if (suggestions.length === 0) {
            this.hideSuggestions();
            return;
        }

        container.innerHTML = suggestions.map(suggestion => 
            `<div class="suggestion-item">${suggestion}</div>`
        ).join('');
        
        container.style.display = 'block';
    }

    // Hide suggestions dropdown
    hideSuggestions() {
        document.getElementById('suggestions').style.display = 'none';
    }

    // Handle suggestion keyboard navigation
    handleSuggestionNavigation(e) {
        const suggestions = document.querySelectorAll('.suggestion-item');
        const currentIndex = Array.from(suggestions).findIndex(item => item.classList.contains('selected'));
        
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.selectSuggestion(currentIndex + 1, suggestions);
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.selectSuggestion(currentIndex - 1, suggestions);
                break;
            case 'Enter':
                if (currentIndex >= 0) {
                    e.preventDefault();
                    const selectedSuggestion = suggestions[currentIndex];
                    if (selectedSuggestion) {
                        document.getElementById('searchQuery').value = selectedSuggestion.textContent;
                        this.hideSuggestions();
                        this.performSearch();
                    }
                }
                break;
            case 'Escape':
                this.hideSuggestions();
                break;
        }
    }

    // Select suggestion with keyboard
    selectSuggestion(index, suggestions) {
        suggestions.forEach(item => item.classList.remove('selected'));
        
        if (index >= 0 && index < suggestions.length) {
            suggestions[index].classList.add('selected');
        }
    }

    // Perform search
    async performSearch() {
        const query = document.getElementById('searchQuery').value.trim();
        const category = document.getElementById('categoryFilter').value;
        const minFollowers = document.getElementById('minFollowers').value;
        const maxFollowers = document.getElementById('maxFollowers').value;

        if (!query) {
            this.showError('Please enter a search query');
            return;
        }

        this.currentSearchQuery = query;
        this.showLoading();
        this.hideSuggestions();

        try {
            const searchData = {
                query: query,
                limit: 10,
                category: category || null,
                min_followers: minFollowers ? parseInt(minFollowers) : null,
                max_followers: maxFollowers ? parseInt(maxFollowers) : null
            };

            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(searchData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayResults(data);

        } catch (error) {
            console.error('Search failed:', error);
            this.showError('Search failed. Please try again.');
        }
    }

    // Display search results
    displayResults(data) {
        this.hideLoading();
        this.hideError();
        this.hideNoResults();

        const resultsContainer = document.getElementById('resultsContainer');
        const resultsList = document.getElementById('resultsList');
        const resultsCount = document.getElementById('resultsCount');
        const searchTime = document.getElementById('searchTime');

        if (data.results.length === 0) {
            this.showNoResults();
            return;
        }

        // Update results info
        resultsCount.textContent = data.total_count;
        searchTime.textContent = `Search completed in ${data.processing_time_ms.toFixed(0)}ms`;

        // Generate results HTML
        const resultsHTML = data.results.map((influencer, index) => 
            this.createInfluencerCard(influencer, index)
        ).join('');

        resultsList.innerHTML = resultsHTML;
        resultsContainer.style.display = 'block';

        // Add fade-in animation
        setTimeout(() => {
            const cards = document.querySelectorAll('.influencer-card');
            cards.forEach((card, index) => {
                setTimeout(() => {
                    card.classList.add('fade-in');
                }, index * 100);
            });
        }, 100);
    }

    // Create influencer card HTML
    createInfluencerCard(influencer, index) {
        const verificationBadge = influencer.is_verified ? 
            '<i class="fas fa-check-circle verification-badge" title="Verified"></i>' : '';
        
        const privacyBadge = influencer.is_private ? 
            '<i class="fas fa-lock privacy-badge" title="Private Account"></i>' : '';

        const username = influencer.username ? 
            `<div class="influencer-username">@${influencer.username}${verificationBadge}${privacyBadge}</div>` : '';

        const stats = [];
        if (influencer.follower_count) {
            stats.push(`<span class="influencer-followers">${this.formatNumber(influencer.follower_count)} followers</span>`);
        }
        if (influencer.post_count) {
            stats.push(`<span>${this.formatNumber(influencer.post_count)} posts</span>`);
        }

        const statsHTML = stats.length > 0 ? 
            `<div class="influencer-stats">${stats.join(' • ')}</div>` : '';

        const actions = [];
        if (influencer.instagram_url) {
            actions.push(`<a href="${influencer.instagram_url}" target="_blank" class="btn btn-outline-primary btn-sm">
                <i class="fas fa-external-link-alt me-1"></i>Instagram
            </a>`);
        }
        actions.push(`<button class="btn btn-primary btn-sm" onclick="app.viewInfluencerDetails('${influencer.influencer_id}')">
            <i class="fas fa-eye me-1"></i>View Details
        </button>`);

        return `
            <div class="col-lg-4 col-md-6 col-sm-12">
                <div class="influencer-card">
                    <img src="${influencer.profile_photo_url}" 
                         alt="${influencer.name}" 
                         class="influencer-image"
                         onerror="this.src='https://via.placeholder.com/300x200?text=Profile+Photo'">
                    <div class="influencer-content">
                        <div class="influencer-name">${influencer.name}</div>
                        ${username}
                        <div class="influencer-bio">${influencer.bio}</div>
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="influencer-category">${influencer.category}</span>
                            <span class="influencer-score">${(influencer.similarity_score * 100).toFixed(1)}% match</span>
                        </div>
                        ${statsHTML}
                        <div class="influencer-actions">
                            ${actions.join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Format numbers with commas
    formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }

    // View influencer details (placeholder for future enhancement)
    viewInfluencerDetails(influencerId) {
        // This could open a modal or navigate to a detail page
        alert(`Viewing details for influencer ${influencerId}. This feature will be implemented in the next phase.`);
    }

    // Show loading state
    showLoading() {
        document.getElementById('loadingState').style.display = 'block';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('noResultsState').style.display = 'none';
        document.getElementById('errorState').style.display = 'none';
    }

    // Hide loading state
    hideLoading() {
        document.getElementById('loadingState').style.display = 'none';
    }

    // Show no results state
    showNoResults() {
        document.getElementById('noResultsState').style.display = 'block';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('errorState').style.display = 'none';
    }

    // Hide no results state
    hideNoResults() {
        document.getElementById('noResultsState').style.display = 'none';
    }

    // Show error state
    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorState').style.display = 'block';
        document.getElementById('resultsContainer').style.display = 'none';
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('noResultsState').style.display = 'none';
    }

    // Hide error state
    hideError() {
        document.getElementById('errorState').style.display = 'none';
    }

    // Retry search
    retrySearch() {
        this.performSearch();
    }
}

// Global functions for HTML onclick handlers
function searchExample(query) {
    document.getElementById('searchQuery').value = query;
    app.performSearch();
}

function retrySearch() {
    app.retrySearch();
}

// Initialize the app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new InfluencerDiscoveryApp();
}); 