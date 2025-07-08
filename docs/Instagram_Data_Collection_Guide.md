# Instagram Data Collection Guide

## ⚠️ IMPORTANT LEGAL & ETHICAL NOTICE

This guide is for **educational and research purposes only**. Before using any automated data collection tools with Instagram:

### Legal Considerations
- **Instagram's Terms of Service** prohibit automated data collection
- **Copyright and intellectual property laws** may apply to user content
- **Privacy laws** (GDPR, CCPA, etc.) regulate personal data collection
- **Platform-specific restrictions** may result in account suspension or legal action

### Ethical Guidelines
- **Only collect publicly available data**
- **Respect user privacy and consent**
- **Use reasonable rate limits** to avoid overwhelming Instagram's servers
- **Consider the impact** on users whose data you're collecting
- **Get explicit permission** when possible

### Recommended Alternatives
Before using automated collection, consider these legal alternatives:
1. **Instagram Creator API** - Official API for approved businesses
2. **Manual data collection** - Time-consuming but fully compliant
3. **Third-party services** - Many offer compliant data access
4. **Partner with influencers directly** - Get data with explicit consent

---

## How Our Instagram Collector Works

The `InstagramDataCollector` class uses the `instaloader` library to responsibly gather influencer data that maps to our search system.

### Data Collected
- **Profile Information**: Name, bio, follower count
- **Profile Pictures**: Downloaded and stored locally
- **Content Thumbnails**: First post image for content representation
- **Automatic Categorization**: Based on bio keywords

### Rate Limiting & Safety Features
- **Configurable delays** between requests (default: 2+ seconds)
- **Retry logic** with exponential backoff
- **Error handling** to prevent crashes
- **Logging** for monitoring and debugging
- **Image downloading** with proper error handling

---

## Installation & Setup

### 1. Install Dependencies
The Instagram data collection features are automatically installed with the main project:

```bash
pip install -e .
```

Or install instaloader separately:
```bash
pip install instaloader>=4.14.0
```

### 2. Create Directories
The collector automatically creates necessary directories:
```
data/
├── images/
│   ├── profiles/     # Profile pictures
│   └── content/      # Content thumbnails
├── influencers.csv   # Main data file
└── *.csv            # Category-specific files
```

---

## Usage Examples

### Basic Usage

```python
from app.instagram_scraper import InstagramDataCollector

# Initialize with conservative settings
collector = InstagramDataCollector(
    output_dir="data",
    delay_between_requests=5.0,  # 5 seconds between requests
    max_retries=2
)

# Collect data for specific usernames (replace with real usernames)
usernames = ["example_user1", "example_user2"]
data = collector.collect_influencer_data(usernames, "my_influencers.csv")
```

### Using the Example Script

We've provided a comprehensive example script:

```bash
python examples/collect_instagram_data.py
```

This script provides several options:
1. **Collect from predefined usernames**
2. **Search hashtags for influencers** (advanced)
3. **Load usernames from JSON file**
4. **Create configuration template**

### Configuration File Method

1. Create a configuration file:
```bash
python examples/collect_instagram_data.py  # Option 4
```

2. Edit `data/usernames_template.json`:
```json
{
  "fitness": ["fitness_influencer1", "fitness_influencer2"],
  "beauty": ["beauty_influencer1", "beauty_influencer2"],
  "tech": ["tech_influencer1"],
  "food": ["food_blogger1", "chef_username"],
  "lifestyle": ["lifestyle_blogger1"]
}
```

3. Rename to `usernames_to_collect.json` and run:
```bash
python examples/collect_instagram_data.py  # Option 3
```

---

## Advanced Features

### Hashtag Discovery

⚠️ **Use with extreme caution** - This is more intrusive than direct username collection.

```python
# Search hashtags for potential influencers
found_usernames = collector.search_by_hashtag(
    "fitness", 
    max_posts=20,      # Limit to reduce server load
    min_followers=50000  # Filter by follower count
)

# Review usernames before collecting data
print("Found usernames:", found_usernames)
```

### Custom Categorization

The system automatically categorizes influencers based on keywords in their bio and username:

```python
category = collector.categorize_influencer(
    bio="Fitness enthusiast and personal trainer", 
    username="fitnesscoach"
)
# Returns: InfluencerCategory.FITNESS
```

### Image Handling

Images are automatically downloaded and stored locally:

```python
# Profile pictures saved to: data/images/profiles/
# Content thumbnails saved to: data/images/content/
# URLs in CSV point to local files for privacy
```

---

## Data Integration

### Loading into Search System

After collecting Instagram data, integrate it with your search system:

```python
from app.data_loader import DataLoader
from app.build_index import main as build_index

# Load the collected data
loader = DataLoader()
data = loader.load_data("data/all_influencers.csv")

# Build search index with new data
build_index()
```

### Data Validation

All collected data is validated against our schema:

```python
from app.schemas import InfluencerData

# Data automatically validated during collection
for record in collected_data:
    influencer = InfluencerData(**record)  # Validates structure
```

---

## Best Practices

### 1. Rate Limiting
```python
collector = InstagramDataCollector(
    delay_between_requests=5.0,  # Minimum 2-3 seconds recommended
    max_retries=2                # Don't be too aggressive
)
```

### 2. Error Handling
```python
try:
    data = collector.get_profile_data(username)
    if data is None:
        print(f"Failed to collect data for {username}")
except Exception as e:
    print(f"Error: {e}")
```

### 3. Logging & Monitoring
```python
import logging
logging.basicConfig(level=logging.INFO)

# The collector automatically logs:
# - Request attempts and failures
# - Rate limiting delays
# - Collection summaries
```

### 4. Data Privacy
- **Store data securely** with appropriate access controls
- **Delete data** when no longer needed
- **Anonymize data** where possible
- **Respect user requests** for data deletion

---

## Troubleshooting

### Common Issues

**1. Connection Errors**
```
Solution: Check internet connection, reduce request frequency
```

**2. Rate Limiting / Blocking**
```
Solution: Increase delay_between_requests, reduce batch sizes
```

**3. Profile Not Found**
```
Solution: Verify username exists, check for typos
```

**4. Image Download Failures**
```
Solution: Check image URLs, ensure sufficient disk space
```

### Debugging Tips

1. **Enable verbose logging**:
```python
logging.getLogger().setLevel(logging.DEBUG)
```

2. **Test with single username first**:
```python
data = collector.get_profile_data("single_username")
```

3. **Check network connectivity**:
```python
import requests
response = requests.get("https://www.instagram.com")
print(f"Status: {response.status_code}")
```

---

## Legal Compliance Checklist

Before using Instagram data collection:

- [ ] **Read Instagram's Terms of Service** - Understand current restrictions
- [ ] **Review applicable laws** - GDPR, CCPA, local privacy laws
- [ ] **Get explicit permission** - From influencers when possible
- [ ] **Limit data collection** - Only collect what you actually need
- [ ] **Implement data security** - Protect collected data appropriately
- [ ] **Plan for data deletion** - Have a retention and deletion policy
- [ ] **Consider alternatives** - Use official APIs when available
- [ ] **Document your compliance** - Keep records of permissions and procedures

---

## Alternative Data Sources

If Instagram data collection poses too many legal/ethical concerns, consider:

### 1. Instagram Official APIs
- **Instagram Creator API** - For approved business use cases
- **Instagram Graph API** - For managing business accounts

### 2. Third-Party Services
- **Creator.co** - Influencer discovery platform
- **Social Blade** - Public social media statistics
- **Upfluence** - Influencer marketing platform
- **AspireIQ** - Creator relationship management

### 3. Manual Collection
- **Direct outreach** - Contact influencers for data
- **Public information** - Use only publicly displayed data
- **Surveys** - Ask influencers to provide their own data

### 4. Partner Programs
- **Influencer networks** - Join existing communities
- **Agency partnerships** - Work with established agencies
- **Creator programs** - Platform-specific creator tools

---

## Support & Resources

### Documentation
- [Instaloader Documentation](https://instaloader.github.io/)
- [Instagram Terms of Service](https://help.instagram.com/581066165581870)
- [GDPR Compliance Guide](https://gdpr.eu/)

### Legal Resources
- Consult with legal counsel familiar with social media law
- Review platform-specific developer policies
- Understand regional privacy regulations

### Technical Support
- Check project issues on GitHub
- Review error logs in `logs/` directory
- Test with small datasets first

---

**Remember: When in doubt, don't collect the data. It's better to have a smaller, compliant dataset than to face legal issues.** 