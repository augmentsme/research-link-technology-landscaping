import config
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions
from itertools import batched
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.decomposition import PCA

class ResearchDBClient:
    def __init__(self):
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key="test",
            model_name=config.db.model,
            api_base=config.db.api_base
        )

        self.client = PersistentClient(path=config.db.path)
        self.grants_collection = self.client.get_or_create_collection(name="grants", embedding_function=self.ef)
        self.keywords_collection = self.client.get_or_create_collection(name="keywords", embedding_function=self.ef)
        self.categories_proposal_collection = self.client.get_or_create_collection(name="categories_proposal", embedding_function=self.ef)
        self.categories_collection = self.client.get_or_create_collection(name="categories", embedding_function=self.ef)
        self.max_batch_size = self.client.get_max_batch_size()

    def batch_add(self, collection, ids, documents):
        for id_batch, doc_batch in zip(batched(ids, self.max_batch_size), batched(documents, self.max_batch_size)):
            collection.add(
                ids=list(id_batch),
                documents=list(doc_batch),
            )
            
    def batch_update_metadata(self, collection, ids, documents, metadatas):
        for id_batch, doc_batch in zip(batched(ids, self.max_batch_size), batched(documents, self.max_batch_size)):
            collection.update(
                ids=list(id_batch),
                documents=list(doc_batch),
                metadatas=list(metadatas)
            )

    def add_grants(self):
        grants = config.Grants.load()
        ids = grants.id.tolist()
        documents = grants.apply(config.Grants.template, axis=1).tolist()
        self.batch_add(self.grants_collection, ids, documents)

    def add_keywords(self):
        keywords = config.Keywords.load()
        ids = keywords.term.tolist()
        documents = keywords.apply(config.Keywords.template, axis=1).tolist()
        self.batch_add(self.keywords_collection, ids, documents)

    def add_categories(self):
        categories = config.Categories.load()
        ids = categories.name.tolist()
        documents = categories.apply(config.Categories.template, axis=1).tolist()
        self.batch_add(self.categories_collection, ids, documents)

    def add_category_proposals(self):
        category_proposals = config.Categories.load_proposal()
        ids = [str(i) for i in range(len(category_proposals))]
        documents = category_proposals.apply(config.Categories.template, axis=1).tolist()
        self.batch_add(self.categories_proposal_collection, ids, documents)


    def cluster_category_proposal(self):
        embeddings = self.categories_proposal_collection.get(include=['embeddings'])['embeddings']
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(embeddings)
        return dbscan

# client = ResearchDBClient()


# from datamapplot import create_plot


# embeddings = client.categories_proposal_collection.get(include=['embeddings'])
# clusterer = KMeans(n_clusters=10, random_state=0)
# clusterer.fit(embeddings['embeddings'])
# labels = clusterer.labels_
# pca = PCA(n_components=2)
# coordinates = pca.fit_transform(embeddings['embeddings'])

# create_plot(coordinates)

# embeddings['embeddings'].shape