#!/bin/bash

sqlite3 vector_store/processed_documents.db < sql_scripts/show_documents.sql
