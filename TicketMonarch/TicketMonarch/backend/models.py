from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import pandas as pd

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False)
    product_name = Column(String(200), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    order_date = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert Order object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'customer_name': self.customer_name,
            'email': self.email,
            'product_name': self.product_name,
            'quantity': self.quantity,
            'price': self.price,
            'total': self.total,
            'order_date': self.order_date.isoformat() if self.order_date else None
        }

# Database setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATA_DIR, 'ticketmonarch.db')
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database and create tables"""
    Base.metadata.create_all(bind=engine)

class Checkout(Base):
    __tablename__ = 'checkouts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True)
    card_number = Column(String(20), nullable=True)
    card_expiry = Column(String(5), nullable=True)
    card_cvv = Column(String(4), nullable=True)
    billing_address = Column(String(200), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    zip_code = Column(String(10), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert Checkout object to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'full_name': self.full_name,
            'email': self.email,
            'card_number': self.card_number,
            'card_expiry': self.card_expiry,
            'card_cvv': self.card_cvv,
            'billing_address': self.billing_address,
            'city': self.city,
            'state': self.state,
            'zip_code': self.zip_code,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

def get_db():
    """Get database session with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_checkout_to_db(checkout_data):
    """Save checkout order data to database"""
    db = SessionLocal()
    try:
        checkout = Checkout(
            full_name=checkout_data.get('full_name', ''),
            email=checkout_data.get('email', ''),
            card_number=checkout_data.get('card_number', ''),
            card_expiry=checkout_data.get('card_expiry', ''),
            card_cvv=checkout_data.get('card_cvv', ''),
            billing_address=checkout_data.get('billing_address', ''),
            city=checkout_data.get('city', ''),
            state=checkout_data.get('state', ''),
            zip_code=checkout_data.get('zip_code', '')
        )
        db.add(checkout)
        db.commit()
        db.refresh(checkout)
        return checkout
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def export_checkouts_to_csv(csv_path=None):
    """
    Export all checkout records from database to CSV format
    
    Args:
        csv_path (str, optional): Path to save CSV file. 
                                 If None, saves to data/checkouts.csv
                                 
    Returns:
        str: Path to the created CSV file
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, 'checkouts.csv')
    
    db = SessionLocal()
    try:
        checkouts = db.query(Checkout).all()
        
        if not checkouts:
            # Create empty CSV with headers if no data
            df = pd.DataFrame(columns=[
                'id', 'full_name', 'email', 'card_number', 'card_expiry',
                'card_cvv', 'billing_address', 'city', 'state', 'zip_code', 'timestamp'
            ])
            df.to_csv(csv_path, index=False)
            return csv_path
        
        # Convert to list of dictionaries
        data = [checkout.to_dict() for checkout in checkouts]
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path
    finally:
        db.close()

def import_checkouts_from_csv(csv_path=None, skip_duplicates=True):
    """
    Import checkout records from CSV file to database
    
    Args:
        csv_path (str, optional): Path to CSV file. 
                                 If None, reads from data/checkouts.csv
        skip_duplicates (bool): If True, skip records that already exist (by email and timestamp)
                                
    Returns:
        tuple: (imported_count, skipped_count, errors)
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, 'checkouts.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    db = SessionLocal()
    imported_count = 0
    skipped_count = 0
    errors = []
    
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['full_name', 'email', 'card_number', 'card_expiry', 
                          'card_cvv', 'billing_address', 'city', 'state', 'zip_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        for index, row in df.iterrows():
            try:
                # Check for duplicates if skip_duplicates is True
                if skip_duplicates:
                    # Parse timestamp if it exists
                    timestamp = None
                    if 'timestamp' in row and pd.notna(row['timestamp']):
                        try:
                            timestamp = pd.to_datetime(row['timestamp'])
                        except:
                            pass
                    
                    # Check if record already exists
                    existing = db.query(Checkout).filter(
                        Checkout.email == row['email'],
                        Checkout.card_number == str(row['card_number'])
                    ).first()
                    
                    if existing:
                        skipped_count += 1
                        continue
                
                # Create new checkout record
                checkout = Checkout(
                    full_name=str(row['full_name']),
                    email=str(row['email']),
                    card_number=str(row['card_number']),
                    card_expiry=str(row['card_expiry']),
                    card_cvv=str(row['card_cvv']),
                    billing_address=str(row['billing_address']),
                    city=str(row['city']),
                    state=str(row['state']),
                    zip_code=str(row['zip_code'])
                )
                
                # Set timestamp if provided
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    try:
                        checkout.timestamp = pd.to_datetime(row['timestamp'])
                    except:
                        pass  # Use default timestamp
                
                db.add(checkout)
                imported_count += 1
                
            except Exception as e:
                errors.append(f"Row {index + 2}: {str(e)}")  # +2 for header and 0-based index
        
        db.commit()
        return imported_count, skipped_count, errors
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

