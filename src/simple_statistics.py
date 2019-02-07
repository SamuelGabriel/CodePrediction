import os
import sys
from glob import glob

from graph_pb2 import Graph
from graph_pb2 import FeatureNode

def main(path):
  total_tokens = 0
  total_lines = 0
  total_javadocs = 0
  for p in glob(f"{path}/**/*.proto", recursive=True):
    with open(p, "rb") as f:
      g = Graph()
      g.ParseFromString(f.read())
      total_tokens += len(list(filter(lambda n:n.type in 
            (FeatureNode.TOKEN,
              FeatureNode.IDENTIFIER_TOKEN), g.node)))
      total_lines += g.ast_root.endLineNumber
      total_javadocs += sum(n.type == FeatureNode.COMMENT_JAVADOC for n in g.node)
  print("%s contains %d, %d, %d" % (path, total_tokens, total_lines, total_javadocs))

if __name__ == "__main__":
  main(sys.argv[1])

