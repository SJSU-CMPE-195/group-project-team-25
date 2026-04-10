export const mockConcerts = [
  {
    id: 1,
    name: "Chappell Roan",
    date: "Feb 14, 2026",
    eventName: "Midwest Princess Tour",
    venue: "Aragon Ballroom",
    city: "Chicago, IL",
    location: "Chicago, IL • Aragon Ballroom",
    image: "/chappellImg.jpg",
    price: 100,
  },
  {
    id: 2,
    name: "Metallica",
    date: "Jun 16, 2026",
    eventName: "M72 World Tour",
    venue: "AT&T Stadium",
    city: "Dallas, TX",
    location: "Dallas, TX • AT&T Stadium",
    image: "/metallicaImg.jpg",
    price: 250,
  },
  {
    id: 3,
    name: "Lady Gaga",
    date: "Oct 16, 2026",
    eventName: "From Zero Tour",
    venue: "Dodger Stadium",
    city: "Los Angeles, CA",
    location: "Los Angeles, CA • Dodger Stadium",
    image: "/gagaImg.jpg",
    price: 200,
  },
];

export const mockConcertsById = Object.fromEntries(
  mockConcerts.map((concert) => [concert.id, concert])
);