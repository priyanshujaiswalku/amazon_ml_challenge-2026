"""
Unit Tests for Phase 2: Domain-Aware Data Cleaning & Multilingual Normalization
================================================================================
Tests legal suffix canonicalization, core name extraction, address abbreviation
expansion, diacritics stripping, and open-set country preservation across US,
India, and France records.
"""

import unittest
import pandas as pd

from src.data_cleaning import (
    clean_business_name,
    clean_business_address,
    clean_country,
    clean_dataframe,
    strip_accents,
)


class TestDataCleaningPhase2(unittest.TestCase):
    """Test suite for Phase 2 cleaning and normalization routines."""

    def test_strip_accents(self):
        """Test unicode diacritic stripping for French and European accents."""
        self.assertEqual(strip_accents("Société"), "Societe")
        self.assertEqual(strip_accents("Générale"), "Generale")
        self.assertEqual(strip_accents("L'Oréal"), "L'Oreal")
        self.assertEqual(strip_accents("Café"), "Cafe")
        self.assertEqual(strip_accents("Allée"), "Allee")
        self.assertEqual(strip_accents("Hypermarché"), "Hypermarche")

    def test_legal_suffix_canonicalization_us(self):
        """Test canonicalization of US legal corporate forms."""
        clean, core = clean_business_name("Amazon Web Services, Inc.")
        self.assertEqual(clean, "amazon web services inc")
        self.assertEqual(core, "amazon web services")

        clean, core = clean_business_name("Microsoft Corporation")
        self.assertEqual(clean, "microsoft corp")
        self.assertEqual(core, "microsoft")

        clean, core = clean_business_name("Alphabet Google LLC")
        self.assertEqual(clean, "alphabet google llc")
        self.assertEqual(core, "alphabet google")

    def test_legal_suffix_canonicalization_india(self):
        """Test canonicalization of Indian business forms."""
        clean, core = clean_business_name("Tata Consultancy Services Pvt. Ltd.")
        self.assertEqual(clean, "tata consultancy services pvt ltd")
        self.assertEqual(core, "tata consultancy services")

        clean, core = clean_business_name("Reliance Industries Private Limited")
        self.assertEqual(clean, "reliance industries pvt ltd")
        self.assertEqual(core, "reliance industries")

    def test_legal_suffix_canonicalization_france(self):
        """Test canonicalization of French legal forms (SARL, SAS, SA, SNC, EURL)."""
        clean, core = clean_business_name("Société Générale SA")
        self.assertEqual(clean, "societe generale sa")
        self.assertEqual(core, "societe generale")

        clean, core = clean_business_name("L'Oréal S.A.S.")
        self.assertEqual(clean, "loreal sas")
        self.assertEqual(core, "loreal")

        clean, core = clean_business_name("Carrefour Hypermarché EURL")
        self.assertEqual(clean, "carrefour hypermarche eurl")
        self.assertEqual(core, "carrefour hypermarche")

        clean, core = clean_business_name("Café de Flore SNC")
        self.assertEqual(clean, "cafe de flore snc")
        self.assertEqual(core, "cafe de flore")

    def test_brand_noise_and_punctuation(self):
        """Test handling of apostrophes, hyphens, and symbol noise."""
        clean, core = clean_business_name("Ben & Jerry's, Inc.")
        self.assertEqual(clean, "ben and jerrys inc")
        self.assertEqual(core, "ben and jerrys")

        clean, core = clean_business_name("Wal-Mart Stores")
        self.assertEqual(clean, "walmart stores")
        self.assertEqual(core, "walmart stores")

        clean, core = clean_business_name("Acme Robotics Incorporated & Co.")
        self.assertEqual(clean, "acme robotics inc and co")
        self.assertEqual(core, "acme robotics")

    def test_address_normalization_us(self):
        """Test US address standardization (street types, unit numbers, states)."""
        addr = clean_business_address("410 Terry Ave N, Suite 500, Seattle, Washington 98109")
        self.assertIn("ave", addr)
        self.assertIn("ste 500", addr)
        self.assertIn("wa 98109", addr)

    def test_address_normalization_india(self):
        """Test Indian address standardization (landmarks, colonies, PINs)."""
        addr = clean_business_address("Opposite Bus Stand, Near Metro Station, Sector 14, Gurugram 122001")
        self.assertIn("opp", addr)
        self.assertIn("nr", addr)
        self.assertIn("sec 14", addr)
        self.assertIn("122001", addr)

    def test_address_normalization_france(self):
        """Test French address standardization (rue, bd, ave, 5-digit postal code)."""
        addr = clean_business_address("29 Boulevard Haussmann, 75009 Paris")
        self.assertEqual(addr, "29 blvd haussmann 75009 paris")

        addr2 = clean_business_address("14 Rue Royale, 75008 Paris")
        self.assertEqual(addr2, "14 rue royale 75008 paris")

    def test_clean_country_preservation(self):
        """Test country normalization and open-set preservation."""
        self.assertEqual(clean_country("US"), "US")
        self.assertEqual(clean_country("United States"), "US")
        self.assertEqual(clean_country("IN"), "IN")
        self.assertEqual(clean_country("India"), "IN")
        self.assertEqual(clean_country("FR"), "FR")
        self.assertEqual(clean_country("France"), "FR")
        # Open-set test: unknown countries must not be destroyed or converted to US/IN
        self.assertEqual(clean_country("DE"), "DE")
        self.assertEqual(clean_country("JP"), "JP")
        self.assertEqual(clean_country(None), "")

    def test_clean_dataframe(self):
        """Test vectorized clean_dataframe method on a sample DataFrame."""
        df = pd.DataFrame({
            "entity_id": ["S1-001", "S1-002"],
            "business_name": ["Société Générale SA", "Tata Consultancy Services Pvt. Ltd."],
            "business_address": ["29 Boulevard Haussmann, 75009 Paris", "Raveline St, Fort, Mumbai 400001"],
            "country": ["France", "India"],
        })
        cleaned_df = clean_dataframe(df)
        self.assertIn("clean_name", cleaned_df.columns)
        self.assertIn("core_name", cleaned_df.columns)
        self.assertIn("clean_address", cleaned_df.columns)
        self.assertIn("clean_country", cleaned_df.columns)
        self.assertEqual(cleaned_df.loc[0, "clean_country"], "FR")
        self.assertEqual(cleaned_df.loc[1, "clean_country"], "IN")
        self.assertEqual(cleaned_df.loc[0, "core_name"], "societe generale")


if __name__ == "__main__":
    unittest.main()
