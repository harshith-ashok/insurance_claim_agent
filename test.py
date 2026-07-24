from query import diagnostic_pipeline

# --- COMPREHENSIVE STRESS-TESTING MATRIX ---

# Easy Semantic
diagnostic_pipeline(query="What are the key benefits of the policy?")

# # Hard Lexical
# diagnostic_pipeline(query="ionising radiations contamination radioactivity nuclear waste")

# # Numerical
# diagnostic_pipeline(query="What is the maximum compensation available?")

# # Section Filter
# diagnostic_pipeline(
#     query="What are the exclusions?",
#     target_section="General Exceptions"
# )

# # Multi-hop
# diagnostic_pipeline(
#     query="What happens to life cover after a critical illness claim is paid?"
# )

# # Long-tail Clause
# diagnostic_pipeline(
#     query="owner-driver holds effective driving license rule 3 central motor vehicles rules"
# )

# # Example Retrieval
# diagnostic_pipeline(
#     query="What happened to Priya after her cervical cancer diagnosis?"
# )

# # Process Retrieval
# diagnostic_pipeline(
#     query="What documents are required for accidental death claims?"
# )
