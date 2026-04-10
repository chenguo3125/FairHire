"""
Full-resume text fixtures for integration tests and demos.

Multi-paragraph strings mimic real CV sections (summary, experience bullets, education).
"""

from __future__ import annotations

# Technical-only: minimal demographic cues; good for near-zero fairness delta checks.
RESUME_TECHNICAL_ONLY = """
ALEX RIVERA
Software Engineer

SUMMARY
Backend engineer with 6 years building distributed systems and observability.

EXPERIENCE
• Cut p99 latency from 800ms to 120ms using an in-memory cache layer in Go.
• Shipped a REST API behind Kubernetes with Prometheus metrics and Grafana dashboards.
• Implemented idempotent workers with PostgreSQL and Redis for job processing.

EDUCATION
B.S. Computer Science — coursework in algorithms and operating systems.
""".strip()

# Mixed: names, affinity groups, socioeconomic + strong technical bullets (typical fairness-audit scenario).
RESUME_MIXED_DEMOGRAPHIC = """
JORDAN SMITH
Product Engineer

SUMMARY
Full-stack engineer focused on reliability and inclusive team practices.

EXPERIENCE
• Served as Vice President of the Women in Computing chapter on campus.
• Received a merit scholarship while completing a capstone on MIPS assembly optimization.
• Built TypeScript microservices on AWS Lambda; improved API availability to 99.95%.
• Mentored interns on unit testing and CI with GitHub Actions.

EDUCATION
M.S. Computer Science — thesis on distributed consensus.
Volunteer with the Asian American Association tech outreach program.
""".strip()
