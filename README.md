## MovieConnect README
## Overview
MovieConnect is a movie recommendation system developed using the PALM2 Large Language Model (LLM) and deployed using Streamlit. 
This innovative system leverages advanced machine learning techniques to analyze user preferences and historical viewing data, delivering personalized movie recommendations.


<img width="774" alt="Screenshot 2023-12-04 at 4 59 47 PM" src="https://github.com/shashankks0709/MovieConnect/assets/71184502/c2fba68c-233c-42c3-abf5-fc57179cb1bc">

## Data sets
tmdb_5000_credits.csv : https://drive.google.com/file/d/1C7aTg8tU9KHMVm5yIA0Gn3nJ4LsdpGMI/view?usp=sharing

tmdb_5000_movies.csv : https://drive.google.com/file/d/14stHfhXpZ8dBQgX4JWCcrbrhbpD1d-ym/view?usp=sharing

## Python libraries:
os

google.generativeai (You need an API key for this)

ipywidgets

pandas

IPython

streamlit 

## Development Process
Using PALM2 LLM
Model Selection: The project utilizes the PALM2 LLM for its robust natural language processing capabilities.

Data Processing: We collected, cleaned, and normalized data to suit the needs of the PALM2 model.

Recommendation Logic: The system applies collaborative filtering or content-based filtering techniques to generate tailored recommendations.

## Setup

Set your PALM API key: You can either set it as an environment variable or directly in the code.

python Copy code: PALM_API_KEY = os.getenv("PALM_API_KEY", "YOUR_API_KEY_HERE")

Load the movie dataset

Create an instance of the Recommend_movies class.

Use the generate method to get movie recommendations.

The interface uses IPython widgets for a simple GUI where users can input a movie name and get recommendations.
Workflow

The Recommend_movies class configures the generative AI model with the specified parameters.
Upon providing a movie name and pressing the button, the model takes a sample prompt with movie names and their corresponding recommendations.
Based on this prompt, the model tries to predict recommendations for the input movie.
The generated recommendations are then cross-referenced with the dataset to ensure they are valid.
The results are displayed using IPython.


## User Interface with Streamlit
Streamlit Integration: The user interface is built and managed using Streamlit, ensuring a user-friendly experience.

Interactive Features: The application allows users to input their preferences and receive recommendations in real-time.
## Deployment
The MovieConnect system is deployed using Streamlit, making it accessible as a web application. This deployment strategy ensures ease of access and use, catering to a broad audience.

## How to Run
Ensure you have Python and Streamlit installed.

Clone the repository: git clone the repo

Navigate to the directory containing app.py.

Run the command: streamlit run app.py.

## Future Improvements
Expanding the dataset for a broader range of recommendations.

Extending the application to mobile devices for wider reach and convenience.

Exploring other LLMs to enhance accuracy and efficiency.


## Conclusion

This recommendation system provides a quick and easy way to find similar movies. With the potential to integrate more advanced features and datasets, this system can be further enhanced to suit various user needs.
