import logo from './logo.svg';
import './App.css';
import { useState } from 'react';
import Button from '@mui/material/Button';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [responseData, setResponseData] = useState('Your output will show up here');
  const [articles, setArticles] = useState([]); 
  const [selectedArticle, setSelectedArticle] = useState(""); 

  const handleClick = (ticker) => {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", "http://localhost:5000/" + ticker);

    xhr.onload = () => {
      if (xhr.status == 200) {
        const response = JSON.parse(xhr.responseText);
        setResponseData(`${ticker}'s sentiment is currently ${response.sentiment}`);

        setArticles(response.articles);

        console.log(response.articles);
      } else {
        console.error("Error with the reqest: " + xhr.responseText);
      }
    };

    xhr.send()
  }

  return (
    <div className="App">
      <div className="container">
        <div className="center">
          <input type="text" className="textbox" placeholder="Enter text here..." onChange={(e) => setInputValue(e.target.value)} />
          <Button className="btn" variant='contained' onClick={() => handleClick(inputValue)}>Search</Button>
        </div>
        <p className='output'>{responseData}</p>

        {articles.length > 0 && (
          <div className='dropdown-container'>
            <label className='dropdown-container'>Articles That Were Used: </label>
            <select className='dropdown' onChange={(e) => setSelectedArticle(e.target.value)}>
              <option value="">-- Choose an article --</option>
              {articles.map((article, index) => (
                <option key={index} value={article}>
                  Article {index + 1}
                </option>
              ))}
            </select>

            {/* Show Selected Article */}
            {selectedArticle && <p className="article-content">{selectedArticle}</p>}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
