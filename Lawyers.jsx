import React, { useState, useEffect } from 'react';
import { FaSearch, FaMicrophone, FaSpellCheck } from 'react-icons/fa';
import LawyerCard from './LawyerCard';

const Lawyers = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [caseDescription, setCaseDescription] = useState('');
  const [lawyers, setLawyers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedSpecialization, setSelectedSpecialization] = useState('All');
  const [selectedLocation, setSelectedLocation] = useState('All');
  const [selectedLanguage, setSelectedLanguage] = useState('All');
  const [selectedExperience, setSelectedExperience] = useState('All');
  const [selectedPrice, setSelectedPrice] = useState('All');
  const [isListening, setIsListening] = useState(false);

  // Initialize speech recognition
  const [recognition, setRecognition] = useState(null);

  const cleanText = (text) => {
    // Remove extra spaces
    text = text.replace(/\s+/g, ' ');
    
    // Fix common punctuation issues
    text = text.replace(/\s+([.,!?])/g, '$1'); // Remove space before punctuation
    text = text.replace(/([.,!?])\s*/g, '$1 '); // Add space after punctuation
    
    // Capitalize first letter of sentences
    text = text.replace(/(^|\.\s+)([a-z])/g, (match, p1, p2) => p1 + p2.toUpperCase());
    
    // Fix common spelling variations
    const spellingCorrections = {
      'crimnal': 'criminal',
      'civilian': 'civil',
      'famly': 'family',
      'corprate': 'corporate',
      'proprty': 'property',
      'intellectualproperty': 'intellectual property',
      'humanrights': 'human rights'
    };
    
    Object.entries(spellingCorrections).forEach(([wrong, correct]) => {
      const regex = new RegExp(wrong, 'gi');
      text = text.replace(regex, correct);
    });
    
    return text.trim();
  };

  const handleTextChange = (e) => {
    const words = e.target.value.split(/\s+/).filter(word => word.length > 0);
    if (words.length <= 500) {
      // Don't apply cleanText in real-time - let users type normally
      setCaseDescription(e.target.value);
    }
  };

  const handleSpeechResult = (event) => {
    const transcript = event.results[0][0].transcript;
    const cleanedTranscript = cleanText(transcript);
    setCaseDescription(prev => {
      const newText = prev + ' ' + cleanedTranscript;
      const words = newText.split(/\s+/).filter(word => word.length > 0);
      return words.length <= 500 ? newText : prev;
    });
  };

  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.onresult = handleSpeechResult;

      setRecognition(recognition);
    }
  }, []);

  // Fetch lawyers from lawyers.json
  useEffect(() => {
    const fetchLawyers = async () => {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:5000/lawyers.json');
        if (!response.ok) {
          throw new Error('Failed to fetch lawyers');
        }
        const data = await response.json();
        setLawyers(data);
      } catch (error) {
        console.error('Error fetching lawyers:', error);
        setError('Unable to fetch lawyers. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchLawyers();
  }, []);

  // Fetch recommended lawyers when case description changes
  useEffect(() => {
    const fetchRecommendedLawyers = async () => {
      if (!caseDescription.trim()) {
        // If no case description, show all lawyers
        const response = await fetch('http://localhost:5000/lawyers.json');
        if (response.ok) {
          const allLawyers = await response.json();
          setLawyers(allLawyers);
        }
        return;
      }
      
      setLoading(true);
      try {
        // Get recommended lawyers from the API
        const response = await fetch('http://localhost:5000/api/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: caseDescription,
            latitude: 0,  // You can get these from browser geolocation if needed
            longitude: 0
          })
        });

        if (!response.ok) {
          throw new Error('Failed to get recommendations');
        }

        const recommendedLawyers = await response.json();
        console.log('Recommended lawyers:', recommendedLawyers);
        
        if (recommendedLawyers && recommendedLawyers.length > 0) {
          setLawyers(recommendedLawyers);
        } else {
          // If no recommendations, show all lawyers
          const allResponse = await fetch('http://localhost:5000/lawyers.json');
          if (allResponse.ok) {
            const allLawyers = await allResponse.json();
            setLawyers(allLawyers);
          }
        }
      } catch (error) {
        console.error('Error getting recommendations:', error);
        setError('Unable to get recommendations. Please try again later.');
        // Fallback to showing all lawyers
        try {
          const response = await fetch('http://localhost:5000/lawyers.json');
          if (response.ok) {
            const allLawyers = await response.json();
            setLawyers(allLawyers);
          }
        } catch (fallbackError) {
          console.error('Error fetching all lawyers:', fallbackError);
        }
      } finally {
        setLoading(false);
      }
    };

    // Debounce the API call
    const timer = setTimeout(fetchRecommendedLawyers, 500);
    return () => clearTimeout(timer);
  }, [caseDescription]);

  // Get unique values for filters
  const specializations = ['All', ...new Set(lawyers.flatMap(lawyer => lawyer.speciality))];
  const locations = ['All', ...new Set(lawyers.map(lawyer => lawyer.location))];
  const languages = ['All', ...new Set(lawyers.flatMap(lawyer => lawyer.languages))];
  const experiences = ['All', '0-5 years', '5-10 years', '10-15 years', '15+ years'];
  const prices = ['All', 'Under ₹1000', '₹1000-₹2000', '₹2000-₹3000', '₹3000+'];

  // Filter lawyers based on search term and selected filters
  const filteredLawyers = lawyers.filter(lawyer => {
    const matchesSearch = lawyer.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         lawyer.speciality.some(spec => spec.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesSpecialization = selectedSpecialization === 'All' || 
                                lawyer.speciality.includes(selectedSpecialization);
    const matchesLocation = selectedLocation === 'All' || lawyer.location === selectedLocation;
    const matchesLanguage = selectedLanguage === 'All' || lawyer.languages.includes(selectedLanguage);
    const matchesExperience = selectedExperience === 'All' || 
      (selectedExperience === '0-5 years' && lawyer.experience <= 5) ||
      (selectedExperience === '5-10 years' && lawyer.experience > 5 && lawyer.experience <= 10) ||
      (selectedExperience === '10-15 years' && lawyer.experience > 10 && lawyer.experience <= 15) ||
      (selectedExperience === '15+ years' && lawyer.experience > 15);
    const matchesPrice = selectedPrice === 'All' ||
      (selectedPrice === 'Under ₹1000' && lawyer.price < 1000) ||
      (selectedPrice === '₹1000-₹2000' && lawyer.price >= 1000 && lawyer.price <= 2000) ||
      (selectedPrice === '₹2000-₹3000' && lawyer.price > 2000 && lawyer.price <= 3000) ||
      (selectedPrice === '₹3000+' && lawyer.price > 3000);
    
    return matchesSearch && matchesSpecialization && matchesLocation && 
           matchesLanguage && matchesExperience && matchesPrice;
  });

  const startListening = () => {
    if (recognition) {
      recognition.start();
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold">Our Expert Lawyers</h1>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar */}
          <div className="w-full lg:w-64 bg-gray-100 p-4 rounded-lg">
            <h2 className="text-xl font-semibold mb-4">Filters</h2>
            
            {/* Specialization Filter */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Specialization</h3>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedSpecialization}
                onChange={(e) => setSelectedSpecialization(e.target.value)}
              >
                {specializations.map(spec => (
                  <option key={spec} value={spec}>{spec}</option>
                ))}
              </select>
            </div>

            {/* Location Filter */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Location</h3>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedLocation}
                onChange={(e) => setSelectedLocation(e.target.value)}
              >
                {locations.map(loc => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
              </select>
            </div>

            {/* Language Filter */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Language</h3>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
              >
                {languages.map(lang => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </select>
            </div>

            {/* Experience Filter */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Experience</h3>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedExperience}
                onChange={(e) => setSelectedExperience(e.target.value)}
              >
                {experiences.map(exp => (
                  <option key={exp} value={exp}>{exp}</option>
                ))}
              </select>
            </div>

            {/* Price Filter */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Price Range</h3>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedPrice}
                onChange={(e) => setSelectedPrice(e.target.value)}
              >
                {prices.map(price => (
                  <option key={price} value={price}>{price}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="flex-1">
            <div className="flex flex-col gap-4 mb-8">
              {/* Case Description Text Box with Speech Button */}
              <div className="w-full">
                <div className="border border-gray-200 rounded-lg bg-gray-50">
                  <div className="px-4 py-2 bg-white rounded-b-lg">
                    <div className="flex flex-col gap-2">
                      <div className="flex items-start gap-2">
                        <textarea 
                          id="caseDescription"
                          rows="4"
                          maxLength="500"
                          className="flex-1 px-0 text-sm text-gray-800 bg-white border-0 focus:ring-0"
                          placeholder="Describe your case to find the most suitable lawyers (max 500 words)..."
                          value={caseDescription}
                          onChange={handleTextChange}
                        />
                        <div className="flex flex-col gap-2">
                          <button
                            onClick={startListening}
                            className={`p-2 rounded-full ${isListening ? 'bg-red-500' : 'bg-blue-500'} text-white hover:bg-opacity-80 transition-colors`}
                            title={isListening ? "Listening..." : "Start Speech Input"}
                          >
                            <FaMicrophone className="w-5 h-5" />
                          </button>
                          <button
                            onClick={() => setCaseDescription(cleanText(caseDescription))}
                            className="p-2 rounded-full bg-green-500 text-white hover:bg-opacity-80 transition-colors"
                            title="Clean and format text"
                          >
                            <FaSpellCheck className="w-5 h-5" />
                          </button>
                        </div>
                      </div>
                      <div className="flex justify-between items-center text-xs text-gray-500">
                        <span>
                          {caseDescription.split(/\s+/).filter(word => word.length > 0).length} / 500 words
                        </span>
                        <span className="text-gray-400">
                          Tip: Keep your description clear and concise. Click the spell check button to clean the text.
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Search Bar */}
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search lawyers by name or specialization..."
                  className="w-full p-3 pl-10 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
                {error}
              </div>
            )}

            {/* Loading State */}
            {loading && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                <p className="mt-4 text-gray-600">Loading lawyers...</p>
              </div>
            )}

            {/* Lawyers Grid */}
            {!loading && !error && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredLawyers.length > 0 ? (
                  filteredLawyers.map((lawyer) => (
                    <LawyerCard key={lawyer.id} lawyer={lawyer} />
                  ))
                ) : (
                  <div className="col-span-full text-center py-8">
                    <p className="text-gray-600">
                      {searchTerm.trim() 
                        ? "No lawyers found matching your search criteria. Try adjusting your filters."
                        : "Use the search bar or filters to find lawyers."}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Lawyers; 
