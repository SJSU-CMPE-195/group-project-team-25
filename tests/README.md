run in project root
```
$env:PYTHONPATH = "src"
pytest  --cov=src/TicketMonarch/backend --cov-report=term-missing
```

src/TicketMonarch/frontend
```
npm test
```

run in frontend
```
npm install --save-dev vitest @testing-library/react jsdom
```