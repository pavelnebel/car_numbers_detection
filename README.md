# Service for recognizing car number in the image

## Service functionality:
- Car number recognition in the image

## Architecture:
- Input image processing module
- ResNET - finds a fragment with a car number on the image
- Found Fragment Alignment Module
- Number detector - determines the contents of the car number

## Design implementation:
- Implemented API based on FastAPI
- Production server - uvicorn
- Service running in Docker

## API description:
- Input json format is controlled (pydantic)
- *Functionality will be expanded