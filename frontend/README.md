# Knowledge Base Frontend

A modern React application with Tailwind CSS for managing and querying a knowledge base system. Features a beautiful, responsive interface with drag-and-drop file uploads, semantic search, and document management.

## Features

- **Modern UI/UX**: Clean, responsive design with Tailwind CSS
- **File Upload**: Drag-and-drop interface with progress indicators
- **Semantic Search**: Natural language querying with result filtering
- **Document Management**: View, manage, and delete uploaded documents
- **Real-time Updates**: Live statistics and document list updates
- **Toast Notifications**: User-friendly feedback for all actions
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Technology Stack

- **React 18**: Modern React with hooks and functional components
- **Tailwind CSS**: Utility-first CSS framework for styling
- **Axios**: HTTP client for API communication
- **React Dropzone**: Drag-and-drop file upload functionality
- **React Hot Toast**: Toast notifications
- **Lucide React**: Beautiful, customizable icons
- **React Scripts**: Create React App build tools

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

The application will be available at http://localhost:3000

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

### Backend Connection

The frontend is configured to connect to the backend API at `http://localhost:8000` by default. Make sure the backend is running before using the frontend.

## Project Structure

```
frontend/
├── public/
│   ├── index.html          # Main HTML file
│   └── manifest.json       # PWA manifest
├── src/
│   ├── components/         # React components
│   │   ├── FileUpload.js   # File upload component
│   │   ├── QueryInterface.js # Search interface
│   │   └── DocumentManager.js # Document management
│   ├── api.js             # API service functions
│   ├── App.js             # Main application component
│   ├── index.js           # React entry point
│   └── index.css          # Global styles and Tailwind
├── package.json           # Dependencies and scripts
├── tailwind.config.js     # Tailwind CSS configuration
└── postcss.config.js      # PostCSS configuration
```

## Components

### FileUpload Component

Handles document uploads with drag-and-drop functionality:

- **Drag & Drop**: Visual feedback for file uploads
- **File Validation**: Checks file types and sizes
- **Progress Indicators**: Shows upload and processing status
- **Error Handling**: Displays user-friendly error messages
- **Success Feedback**: Shows upload results and statistics

### QueryInterface Component

Provides semantic search functionality:

- **Search Input**: Multi-line text area for queries
- **Result Configuration**: Adjustable number of results
- **Document Filtering**: Filter search by specific documents
- **Result Display**: Formatted search results with metadata
- **Similarity Scores**: Shows match percentages

### DocumentManager Component

Manages uploaded documents:

- **Document List**: Table view of all uploaded documents
- **Statistics Dashboard**: Shows system statistics
- **Bulk Operations**: Select and delete multiple documents
- **Document Details**: File type, size, upload date, chunk count
- **Refresh Functionality**: Manual refresh of document list

## API Integration

The frontend communicates with the backend through the `api.js` service:

### Available Functions

- `uploadDocument(file)`: Upload a document
- `queryKnowledgeBase(query, nResults, documentIds)`: Search the knowledge base
- `getStats()`: Get system statistics
- `listDocuments()`: Get list of uploaded documents
- `deleteDocuments(documentIds)`: Delete documents
- `healthCheck()`: Check API health

### Error Handling

All API calls include comprehensive error handling:

- **Network Errors**: Connection issues and timeouts
- **Validation Errors**: Invalid file types or data
- **Server Errors**: Backend processing failures
- **User Feedback**: Toast notifications for all error types

## Styling

### Tailwind CSS

The application uses Tailwind CSS for styling with custom configuration:

- **Custom Colors**: Primary and secondary color palettes
- **Custom Animations**: Fade-in and slide-up animations
- **Component Classes**: Reusable button, card, and input styles
- **Responsive Design**: Mobile-first responsive breakpoints

### Custom Components

```css
.btn-primary    /* Primary action buttons */
.btn-secondary  /* Secondary action buttons */
.btn-danger     /* Destructive action buttons */
.card           /* Card containers */
.input-field    /* Form input fields */
.loading-spinner /* Loading indicators */
```

## Development

### Available Scripts

```bash
npm start        # Start development server
npm run build    # Build for production
npm test         # Run tests
npm run eject    # Eject from Create React App
```

### Adding New Features

1. **New Components**: Create in `src/components/`
2. **API Integration**: Add functions to `src/api.js`
3. **Styling**: Use Tailwind classes or add custom CSS
4. **State Management**: Use React hooks for local state

### Code Style

- **Functional Components**: Use hooks instead of class components
- **ES6+ Features**: Use modern JavaScript features
- **Component Composition**: Break down complex components
- **Prop Validation**: Use PropTypes or TypeScript
- **Error Boundaries**: Handle component errors gracefully

## Building for Production

```bash
npm run build
```

This creates a `build` folder with optimized production files.

### Deployment

The build folder can be deployed to any static hosting service:

- **Netlify**: Drag and drop the build folder
- **Vercel**: Connect your repository
- **AWS S3**: Upload build files to S3 bucket
- **GitHub Pages**: Use gh-pages package

## Browser Support

The application supports all modern browsers:

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance

### Optimization Features

- **Code Splitting**: Automatic code splitting by Create React App
- **Lazy Loading**: Components loaded on demand
- **Image Optimization**: Optimized images and icons
- **Bundle Analysis**: Use `npm run build -- --analyze` for bundle analysis

### Best Practices

- **Memoization**: Use React.memo for expensive components
- **Debouncing**: Debounce search inputs
- **Error Boundaries**: Catch and handle errors gracefully
- **Loading States**: Show loading indicators for async operations

## Troubleshooting

### Common Issues

1. **Backend Connection**: Ensure backend is running on port 8000
2. **CORS Errors**: Backend should have CORS enabled
3. **File Upload Issues**: Check file size and type restrictions
4. **Build Errors**: Clear node_modules and reinstall dependencies

### Development Tips

- Use React Developer Tools for debugging
- Check browser console for error messages
- Verify API endpoints with tools like Postman
- Test on different screen sizes for responsiveness

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.