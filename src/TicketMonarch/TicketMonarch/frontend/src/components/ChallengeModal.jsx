import { useState } from 'react'
import './ChallengeModal.css'

const PUZZLES = {
  easy: {
    question: 'What is 3 + 4?',
    answer: '7',
    instruction: 'Please verify you are human.',
  },
  medium: {
    question: 'Type the word shown: MONARCH',
    answer: 'MONARCH',
    instruction: 'Please type the word exactly as shown.',
  },
  hard: {
    question: 'What comes next in the sequence: 2, 4, 8, 16, __?',
    answer: '32',
    instruction: 'Complete the pattern to verify you are human.',
  },
}

function ChallengeModal({ type, difficulty, onComplete }) {
  const [userAnswer, setUserAnswer] = useState('')
  const [attempts, setAttempts] = useState(0)
  const maxAttempts = 3

  if (type === 'blocked') {
    return (
      <div className="challenge-overlay">
        <div className="challenge-modal">
          <h2>Access Denied</h2>
          <p>Our system has detected unusual activity. This checkout has been blocked.</p>
          <button className="challenge-btn" onClick={() => onComplete(false)}>
            Go Back
          </button>
        </div>
      </div>
    )
  }

  const puzzle = PUZZLES[difficulty] || PUZZLES.easy

  const handleSubmit = (e) => {
    e.preventDefault()
    if (userAnswer.trim().toLowerCase() === puzzle.answer.toLowerCase()) {
      onComplete(true)
    } else {
      const newAttempts = attempts + 1
      setAttempts(newAttempts)
      if (newAttempts >= maxAttempts) {
        onComplete(false)
      } else {
        setUserAnswer('')
      }
    }
  }

  return (
    <div className="challenge-overlay">
      <div className="challenge-modal">
        <h2>Verification Required</h2>
        <p>{puzzle.instruction}</p>
        <div className="puzzle-content">
          <p className="puzzle-question">{puzzle.question}</p>
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={userAnswer}
              onChange={(e) => setUserAnswer(e.target.value)}
              placeholder="Your answer"
              autoFocus
            />
            <button type="submit" className="challenge-btn">
              Submit
            </button>
          </form>
          {attempts > 0 && (
            <p className="attempt-warning">
              Incorrect. {maxAttempts - attempts} attempt(s) remaining.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChallengeModal
