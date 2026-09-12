-- MySQL dump 10.13  Distrib 8.0.43, for Win64 (x86_64)
--
-- Host: localhost    Database: F1_strategy
-- ------------------------------------------------------
-- Server version	8.0.43

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `data_sources`
--

DROP TABLE IF EXISTS `data_sources`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `data_sources` (
  `source_id` int NOT NULL AUTO_INCREMENT,
  `source_name` varchar(100) NOT NULL,
  `source_type` enum('Real World','Simulator') NOT NULL,
  `version` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`source_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `drivers`
--

DROP TABLE IF EXISTS `drivers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `drivers` (
  `driver_id` int NOT NULL,
  `driver_code` varchar(3) NOT NULL,
  `driver_name` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`driver_id`),
  UNIQUE KEY `driver_code` (`driver_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `laps`
--

DROP TABLE IF EXISTS `laps`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `laps` (
  `lap_id` int NOT NULL AUTO_INCREMENT,
  `driver_id` int DEFAULT NULL,
  `session_id` int DEFAULT NULL,
  `lap_number` int DEFAULT NULL,
  `lap_time_ms` int DEFAULT NULL,
  `tyre_compound` enum('Hypersoft','Ultrasoft','Supersoft','Soft','Medium','Hard','Superhard','Intermediate','Wet') DEFAULT NULL,
  `tyre_age` int DEFAULT NULL,
  `fuel_load` float DEFAULT NULL,
  `is_valid` tinyint(1) DEFAULT '1',
  -- When this lap was captured live (game UDP capture or a live feed).
  -- NULL for historical FastF1 imports, which are never "live".
  `captured_at` datetime DEFAULT NULL,
  PRIMARY KEY (`lap_id`),
  KEY `session_id` (`session_id`),
  KEY `fk_laps_driver` (`driver_id`),
  KEY `idx_laps_session_lap` (`session_id`, `lap_number`),
  KEY `idx_laps_session_time` (`session_id`, `lap_time_ms`),
  CONSTRAINT `fk_laps_driver` FOREIGN KEY (`driver_id`) REFERENCES `drivers` (`driver_id`),
  CONSTRAINT `laps_ibfk_1` FOREIGN KEY (`session_id`) REFERENCES `sessions` (`session_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7674 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `regulations`
--

DROP TABLE IF EXISTS `regulations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `regulations` (
  `regulation_id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `year_start` int NOT NULL,
  `year_end` int NOT NULL,
  PRIMARY KEY (`regulation_id`)
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `seasons`
--

DROP TABLE IF EXISTS `seasons`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `seasons` (
  `season_id` int NOT NULL AUTO_INCREMENT,
  `year` int NOT NULL,
  `regulation_id` int NOT NULL,
  PRIMARY KEY (`season_id`),
  UNIQUE KEY `year` (`year`),
  KEY `regulation_id` (`regulation_id`),
  CONSTRAINT `seasons_ibfk_1` FOREIGN KEY (`regulation_id`) REFERENCES `regulations` (`regulation_id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sessions`
--

DROP TABLE IF EXISTS `sessions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sessions` (
  `session_id` int NOT NULL AUTO_INCREMENT,
  `driver_id` int DEFAULT NULL,
  `track_name` varchar(50) DEFAULT NULL,
  `session_type` enum('Race','Qualifying','Practice') DEFAULT NULL,
  `weather` varchar(20) DEFAULT NULL,
  `date` date DEFAULT NULL,
  `season_id` int DEFAULT NULL,
  `source_id` int DEFAULT NULL,
  `track_id` int DEFAULT NULL,
  `regulation_id` int DEFAULT NULL,
  PRIMARY KEY (`session_id`),
  KEY `fk_sessions_season` (`season_id`),
  KEY `fk_sessions_source` (`source_id`),
  KEY `fk_sessions_track` (`track_id`),
  KEY `fk_sessions_regulation` (`regulation_id`),
  KEY `fk_sessions_driver` (`driver_id`),
  CONSTRAINT `fk_sessions_driver` FOREIGN KEY (`driver_id`) REFERENCES `drivers` (`driver_id`),
  CONSTRAINT `fk_sessions_regulation` FOREIGN KEY (`regulation_id`) REFERENCES `regulations` (`regulation_id`),
  CONSTRAINT `fk_sessions_season` FOREIGN KEY (`season_id`) REFERENCES `seasons` (`season_id`),
  CONSTRAINT `fk_sessions_source` FOREIGN KEY (`source_id`) REFERENCES `data_sources` (`source_id`),
  CONSTRAINT `fk_sessions_track` FOREIGN KEY (`track_id`) REFERENCES `tracks` (`track_id`)
) ENGINE=InnoDB AUTO_INCREMENT=81 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `strategy_events`
--

DROP TABLE IF EXISTS `strategy_events`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `strategy_events` (
  `event_id` int NOT NULL AUTO_INCREMENT,
  `lap_id` int DEFAULT NULL,
  `event_type` enum('PitStop','SafetyCar','VSC','RedFlag') DEFAULT NULL,
  `duration_sec` float DEFAULT NULL,
  PRIMARY KEY (`event_id`),
  KEY `lap_id` (`lap_id`),
  KEY `idx_events_type_lap` (`event_type`, `lap_id`),
  CONSTRAINT `strategy_events_ibfk_1` FOREIGN KEY (`lap_id`) REFERENCES `laps` (`lap_id`)
) ENGINE=InnoDB AUTO_INCREMENT=260 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `telemetry`
--

DROP TABLE IF EXISTS `telemetry`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `telemetry` (
  `telemetry_id` int NOT NULL AUTO_INCREMENT,
  `lap_id` int DEFAULT NULL,
  `speed` int DEFAULT NULL,
  `throttle` float DEFAULT NULL,
  `brake` float DEFAULT NULL,
  `gear` int DEFAULT NULL,
  `rpm` int DEFAULT NULL,
  `drs` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`telemetry_id`),
  KEY `lap_id` (`lap_id`),
  CONSTRAINT `telemetry_ibfk_1` FOREIGN KEY (`lap_id`) REFERENCES `laps` (`lap_id`)
) ENGINE=InnoDB AUTO_INCREMENT=28384 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `track_aliases`
--

DROP TABLE IF EXISTS `track_aliases`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `track_aliases` (
  `alias_id` int NOT NULL AUTO_INCREMENT,
  `track_id` int NOT NULL,
  `alias` varchar(100) NOT NULL,
  PRIMARY KEY (`alias_id`),
  KEY `track_id` (`track_id`),
  CONSTRAINT `track_aliases_ibfk_1` FOREIGN KEY (`track_id`) REFERENCES `tracks` (`track_id`)
) ENGINE=InnoDB AUTO_INCREMENT=63 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `tracks`
--

DROP TABLE IF EXISTS `tracks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tracks` (
  `track_id` int NOT NULL AUTO_INCREMENT,
  `canonical_name` varchar(100) NOT NULL,
  `country` varchar(100) DEFAULT NULL,
  `short_code` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`track_id`),
  UNIQUE KEY `canonical_name` (`canonical_name`)
) ENGINE=InnoDB AUTO_INCREMENT=61 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-08-14 17:02:29
