# if __name__ == "__main__":

#     config = RunnableConfig(
#         configurable={"doc_id": "Cancom_240514", "collection_id": "Cancom_OCR"}
#     )
#     index_config = IndexConfig.from_runnable_config(config)
#     provider_factory = get_provider_factory_from_config(config)
    
#     embedding_provider, model_name = extract_provider_and_model(
#         index_config.embedding_model
#     )
#     embedding_model = provider_factory.build_embeddings(
#         provider=embedding_provider, model_name=model_name
#     )
    
#     vstore = cast(
#         Chroma, provider_factory.build_vstore(embedding_model, config.vstore, config.collection_id)
#     )
#     vstore.delete_collection()

#     filter = cast(str, {"doc_id": index_config.doc_id})
#     retriever = vstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": k, "filter": filter},
#     )

#     docs = retriever.invoke("Query")

#     pretty_print_docs(docs)
