# Image Structure for Influencer Data

This file explains how images are structured in the influencer discovery tool and how to replace placeholder URLs with real images.

## Image URL Structure

The sample data uses placeholder image URLs that follow this pattern:

### Profile Photos
```
https://images.unsplash.com/photo-{id}?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80
```

### Content Thumbnails  
```
https://images.unsplash.com/photo-{id}?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80
```

## Replacing with Real Images

### Option 1: Local Images
1. Create an `images/` directory in the `data/` folder
2. Save your influencer images in organized subdirectories:
   ```
   data/
   ├── images/
   │   ├── profiles/
   │   │   ├── influencer_001_profile.jpg
   │   │   └── influencer_002_profile.jpg
   │   └── content/
   │       ├── influencer_001_post1.jpg
   │       └── influencer_002_post1.jpg
   ```
3. Update your CSV/database with local paths: `./data/images/profiles/influencer_001_profile.jpg`

### Option 2: Cloud Storage URLs
Replace placeholder URLs with your actual image URLs from:
- AWS S3: `https://your-bucket.s3.amazonaws.com/images/profile_001.jpg`
- Google Cloud Storage: `https://storage.googleapis.com/your-bucket/images/profile_001.jpg`
- Azure Blob Storage: `https://youraccount.blob.core.windows.net/images/profile_001.jpg`

### Option 3: CDN URLs
Use your CDN URLs for faster loading:
- Cloudinary: `https://res.cloudinary.com/your-account/image/upload/v1234567890/profile_001.jpg`
- ImageKit: `https://ik.imagekit.io/your-id/images/profile_001.jpg`

## Image Requirements

- **Supported formats**: JPG, JPEG, PNG, WebP
- **Recommended size**: 400x400px for profiles, 600x400px for content
- **Max file size**: 2MB (configurable in .env)
- **Accessibility**: Ensure images have proper alt text descriptions

## CLIP Embedding Considerations

The CLIP model works best with:
- Clear, well-lit images
- Faces that are visible and not obscured
- Content that clearly shows the described characteristics
- High contrast and good image quality

## Security Notes

- Validate all image URLs before processing
- Implement proper access controls for private images
- Consider image compression for faster loading
- Use HTTPS URLs for security 