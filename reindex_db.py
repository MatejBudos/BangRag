from main import DEFAULT_SOURCE_PATH, create_pipeline, get_files_in_directory


def main() -> None:
    pipeline = create_pipeline()
    document_paths = get_files_in_directory(DEFAULT_SOURCE_PATH)

    if not document_paths:
        print(f"No source documents found in {DEFAULT_SOURCE_PATH}.")
        return

    print("Resetting datastore...")
    pipeline.reset()

    print(f"Indexing {len(document_paths)} source files from {DEFAULT_SOURCE_PATH}...")
    pipeline.add_documents(document_paths)

    try:
        row_count = pipeline.datastore.table.count_rows()
    except Exception:
        row_count = 0

    print(f"Done. Indexed chunks in LanceDB: {row_count}")


if __name__ == "__main__":
    main()
