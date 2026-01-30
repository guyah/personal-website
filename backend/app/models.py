from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Post(Base):
	__tablename__ = "posts"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	title: Mapped[str] = mapped_column(String(200), nullable=False)
	slug: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
	summary: Mapped[str | None] = mapped_column(String(500))
	body: Mapped[str | None] = mapped_column(Text)
	created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
