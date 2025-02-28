from typing import List, Dict, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv("/Users/jayantkapoor/Documents/GitHub/w25-llm-tutorials/.env")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

class BookRecommendation(BaseModel):
    title: str
    author: str
    year: int
    genre: str
    summary: str

def basic_json_parsing():
    """Demonstrate basic JSON parsing with structured output"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides book recommendations."),
        ("user", """Please provide a book recommendation with the following fields:
         title, author, year, genre, and a brief summary.""")
    ])

    # Format the prompt and get LLM response with structured output
    structured_llm = llm.with_structured_output(BookRecommendation)
    formatted_prompt = prompt.format_messages()
    response = structured_llm.invoke(formatted_prompt)
    
    print("\nBasic Structured Output Example:")
    print(response)
    print(f"Recommended Book: {response.title} by {response.author}")

class Movie(BaseModel):
    title: str
    director: str
    year: int
    plot: str

class MovieList(BaseModel):
    movies: List[Movie]

def structured_json_parsing():
    """Demonstrate parsing with array of structured objects"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that provides movie information.
         Always respond with exactly 3 movie recommendations."""),
        ("user", """Give me 3 classic movies from the {genre} genre.
         Each movie should have: title, director, year, and a one-line plot.""")
    ])

    # Format the prompt and get LLM response with structured output
    structured_llm = llm.with_structured_output(MovieList)
    formatted_prompt = prompt.format_messages(genre="film noir")
    response = structured_llm.invoke(formatted_prompt)
    
    print("\nStructured Output Array Example:")
    print(response)
    for movie in response.movies:
        print(f"\nMovie: {movie.title} ({movie.year})")
        print(f"Director: {movie.director}")
        print(f"Plot: {movie.plot}")

class MovieDetails(BaseModel):
    title: str
    year: int
    director: str
    box_office: float

class Character(BaseModel):
    name: str
    actor: str

class Franchise(BaseModel):
    franchise_name: str
    total_movies: int
    movies: List[MovieDetails]
    main_characters: List[Character]

def nested_json_parsing():
    """Demonstrate parsing nested structured objects"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides detailed movie information."),
        ("user", """Provide information about a movie franchise with the following structure:
         - franchise_name
         - total_movies
         - movies (array of movies with title, year, director, box_office)
         - main_characters (array of character names and actors)
         Use the {franchise} franchise.""")
    ])

    # Format the prompt and get LLM response with structured output
    structured_llm = llm.with_structured_output(Franchise)
    formatted_prompt = prompt.format_messages(franchise="The Lord of the Rings")
    response = structured_llm.invoke(formatted_prompt)
    
    print("\nNested Structured Output Example:")
    print(response)
    print(f"\nFranchise: {response.franchise_name}")
    print(f"Number of Movies: {response.total_movies}")
    print("\nMovies:")
    for movie in response.movies:
        print(f"- {movie.title} ({movie.year}) - ${movie.box_office} million")
    print("\nMain Characters:")
    for character in response.main_characters:
        print(f"- {character.name} played by {character.actor}")

def main():
    """Run all structured output examples"""
    basic_json_parsing()
    structured_json_parsing()
    nested_json_parsing()

if __name__ == "__main__":
    main() 