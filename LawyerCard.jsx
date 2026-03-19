import React from 'react';
import { Link } from 'react-router-dom';

const LawyerCard = ({ lawyer }) => {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-gray-800">{lawyer.name}</h3>
          <div className="flex items-center">
            <span className="text-yellow-500 mr-1">★</span>
            <span className="text-gray-600">{lawyer.rating.toFixed(1)}</span>
          </div>
        </div>
        
        <div className="space-y-3">
          <div>
            <p className="text-sm text-gray-500">Specialties</p>
            <div className="flex flex-wrap gap-2 mt-1">
              {lawyer.speciality.map((spec, index) => (
                <span 
                  key={index}
                  className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                >
                  {spec}
                </span>
              ))}
            </div>
          </div>
          
          <div>
            <p className="text-sm text-gray-500">Experience</p>
            <p className="text-gray-800">{lawyer.experience} years</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-500">Location</p>
            <p className="text-gray-800">{lawyer.location}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-500">Languages</p>
            <div className="flex flex-wrap gap-2 mt-1">
              {lawyer.languages.map((lang, index) => (
                <span 
                  key={index}
                  className="px-2 py-1 bg-green-100 text-green-800 text-sm rounded-full"
                >
                  {lang}
                </span>
              ))}
            </div>
          </div>
          
          <div>
            <p className="text-sm text-gray-500">Price</p>
            <p className="text-gray-800">₹{lawyer.price}</p>
          </div>

          <div>
            <p className="text-sm text-gray-500">Phone</p>
            <p className="text-gray-800">{lawyer.phone || 'N/A'}</p>
          </div>

          <div>
            <p className="text-sm text-gray-500">Email</p>
            <p className="text-gray-800">{lawyer.email || 'N/A'}</p>
          </div>
        </div>

        <div className="mt-6 flex justify-between items-center">
          <Link 
            to={`/lawyer/${lawyer.id}`}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors duration-300"
          >
            Request
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LawyerCard; 
