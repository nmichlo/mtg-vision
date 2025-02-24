
# MTG Vision

## Components

1. Data scraping
   - [ ] mtg card image & info scraping
     - [ ] dump images to hdf5
     - [ ] dump info to duckdb/dolt?

2. live view
   - [ ] UI
   - [ ] mtg card detection
     - [ ] detection tracking
     - [ ] camera tracking for detection overlay stabilisation
   - [ ] mtg card recognition
     - [ ] Vector database

3. model training
    - [ ] mtg card detection/segmentation
      - [ ] augmented data generation
      - [ ] model definition
      - [ ] model training
    - [ ] mtg card embedding model (recognition)
      - [ ] augmented data generation
      - [ ] model definition
      - [ ] model training
    - [ ] combined model (yolo v10 with embeddings?)
      - [ ] model definition
      - [ ] model training
