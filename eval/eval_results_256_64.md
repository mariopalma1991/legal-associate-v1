# RAG Retriever Evaluation Results

Generated: 2026-06-16 12:14

## Configuration
- Chunk size : 256 tokens
- Overlap    : 64 tokens
- Eval set   : 189 questions  (specific=100, topic=9, metadata=60, summary=20)

## Overall Results

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Latency (ms) |
|--------|-------|-------|--------|-----|-------------|
| BM25 only | 0.011 | 0.011 | 0.011 | 0.011 | 95 ms |
| Dense - Cohere | 0.185 | 0.312 | 0.407 | 0.255 | 820 ms |
| **Dense - Cohere + Rerank** | 0.206 | 0.407 | 0.455 | 0.292 | 1489 ms |
| Hybrid Cohere + BM25 | 0.185 | 0.317 | 0.413 | 0.256 | 575 ms |
| Hybrid Cohere + Rerank | 0.212 | 0.402 | 0.450 | 0.293 | 1291 ms |

## Hit@5 by Question Type

| Config | Specific | Topic | Metadata | Summary |
|--------|----------|-------|----------|---------|
| BM25 only | 0.000 | 0.222 | 0.000 | 0.000 |
| Dense - Cohere | 0.180 | 0.778 | 0.400 | 0.500 |
| Dense - Cohere + Rerank | 0.310 | 0.889 | 0.450 | 0.550 |
| Hybrid Cohere + BM25 | 0.170 | 0.889 | 0.417 | 0.500 |
| Hybrid Cohere + Rerank | 0.310 | 0.889 | 0.433 | 0.550 |

## Winner: Dense - Cohere + Rerank
Best Hit@5: 0.407
Improvement over BM25: +3750.0%

## Sample Questions by Type

**Specific:** *¿Cuál es la fecha y hora límite para presentar las solicitudes de aclaración sobre las bases de la licitación?*
**Topic:** *¿Qué licitaciones de construcción y pavimentación hay abiertas en Chihuahua?*
**Metadata:** *¿Cuál es el presupuesto estimado o rango de inversión para la rehabilitación del gimnasio y construcción de baños en la secundaria?*
**Summary:** *¿Cuáles son las especificaciones técnicas exactas de los uniformes operativos (materiales, diseño, tallas, cantidades) y cuáles son los términos de entrega, garantía y penalidades por incumplimiento?*