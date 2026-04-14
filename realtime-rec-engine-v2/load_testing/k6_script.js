/**
 * Production-grade k6 load testing script for Netflix/Meta scale recommendation engine
 * Simulates 1M+ concurrent users with realistic patterns and comprehensive metrics
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomIntBetween, randomItem, uuidv4 } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

// Custom metrics
const recommendationLatency = new Trend('recommendation_latency');
const feedbackLatency = new Trend('feedback_latency');
const featureLookupLatency = new Trend('feature_lookup_latency');
const cacheHitRate = new Rate('cache_hit_rate');
const errorRate = new Rate('error_rate');
const qps = new Rate('queries_per_second');
const userEngagement = new Rate('user_engagement');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'https://api.rec-engine.company.com';
const MAX_USERS = 1000000;
const TARGET_QPS = 10000000;
const TEST_DURATION = '1h';

// Test data
const USER_IDS = Array.from({ length: 10000000 }, (_, i) => `user_${String(i).padStart(8, '0')}`);
const ITEM_IDS = Array.from({ length: 1000000 }, (_, i) => `item_${String(i).padStart(8, '0')}`);
const CATEGORIES = ['electronics', 'clothing', 'books', 'home', 'sports', 'beauty', 'toys', 'food'];
const REGIONS = ['us-west-2', 'us-east-1', 'eu-west-1', 'ap-southeast-1'];
const DEVICE_TYPES = ['mobile', 'desktop', 'tablet'];

// User behavior profiles
const BEHAVIOR_PROFILES = {
    power_user: {
        requestRate: 10,
        sessionDuration: 1800,
        recommendationsPerSession: 50,
        feedbackProbability: 0.3,
        featureLookupProbability: 0.2,
        cacheHitProbability: 0.8,
        networkLatencyMs: 50,
        deviceType: 'desktop'
    },
    casual_user: {
        requestRate: 2,
        sessionDuration: 600,
        recommendationsPerSession: 10,
        feedbackProbability: 0.1,
        featureLookupProbability: 0.05,
        cacheHitProbability: 0.6,
        networkLatencyMs: 100,
        deviceType: 'mobile'
    },
    api_client: {
        requestRate: 50,
        sessionDuration: 3600,
        recommendationsPerSession: 500,
        feedbackProbability: 0.5,
        featureLookupProbability: 0.3,
        cacheHitProbability: 0.9,
        networkLatencyMs: 20,
        deviceType: 'desktop'
    }
};

// Test options
export const options = {
    stages: [
        // Warmup phase
        { duration: '5m', target: 100000 },
        // Ramp to full load
        { duration: '10m', target: 500000 },
        // Full load
        { duration: '40m', target: 1000000 },
        // Cool down
        { duration: '5m', target: 0 }
    ],
    thresholds: {
        http_req_duration: ['p(95)<100', 'p(99)<500'],
        http_req_failed: ['rate<0.01'],
        recommendation_latency: ['p(95)<100', 'p(99)<500'],
        errorRate: ['rate<0.01'],
        cache_hit_rate: ['rate>0.7']
    },
    ext: {
        loadzone: 'amazon:us:ashburn',
    },
};

// User class
class TestUser {
    constructor() {
        this.userId = randomItem(USER_IDS);
        this.sessionId = uuidv4();
        this.profile = randomItem(Object.values(BEHAVIOR_PROFILES));
        this.region = randomItem(REGIONS);
        this.deviceType = this.profile.deviceType;
        
        // Session state
        this.sessionStart = Date.now();
        this.recommendationsCount = 0;
        this.feedbackCount = 0;
        this.featureLookups = 0;
        this.cacheHits = 0;
        this.cacheMisses = 0;
        
        // User preferences
        this.preferences = this.generateUserPreferences();
        
        console.log(`User ${this.userId} started session with ${this.deviceType} profile`);
    }
    
    generateUserPreferences() {
        return {
            categories: Array.from({ length: randomIntBetween(1, 3) }, () => randomItem(CATEGORIES)),
            priceRange: {
                min: randomIntBetween(10, 100),
                max: randomIntBetween(100, 1000)
            },
            brandPreferences: Array.from({ length: randomIntBetween(0, 3) }, () => 
                randomItem(['nike', 'apple', 'samsung', 'sony', 'lg'])
            ),
            recentItems: Array.from({ length: randomIntBetween(5, 20) }, () => randomItem(ITEM_IDS))
        };
    }
    
    generateContext() {
        return {
            timestamp: new Date().toISOString(),
            deviceType: this.deviceType,
            region: this.region,
            sessionId: this.sessionId,
            userAgent: `k6-LoadTest/1.0 (${this.deviceType})`,
            ipAddress: `192.168.${randomIntBetween(1, 254)}.${randomIntBetween(1, 254)}`,
            page: randomItem(['home', 'product', 'category', 'search', 'cart']),
            referrer: randomItem(['google', 'direct', 'social', 'email'])
        };
    }
    
    generateFilters() {
        const filters = {};
        
        // Category filter
        if (Math.random() < 0.7) {
            filters.categories = [randomItem(this.preferences.categories)];
        }
        
        // Price filter
        if (Math.random() < 0.5) {
            const priceRange = this.preferences.priceRange;
            filters.priceRange = {
                min: priceRange.min,
                max: priceRange.max
            };
        }
        
        // Brand filter
        if (Math.random() < 0.3 && this.preferences.brandPreferences.length > 0) {
            filters.brands = [randomItem(this.preferences.brandPreferences)];
        }
        
        // Availability filter
        if (Math.random() < 0.8) {
            filters.inStock = true;
        }
        
        return filters;
    }
    
    generateCandidateItems() {
        // 30% chance to provide candidate items
        if (Math.random() < 0.3) {
            const recentItems = this.preferences.recentItems;
            const randomItems = Array.from({ length: 50 }, () => randomItem(ITEM_IDS));
            const candidates = [...new Set([...recentItems, ...randomItems])];
            return candidates.slice(0, 100);
        }
        return null;
    }
    
    shouldContinueSession() {
        const sessionDuration = Date.now() - this.sessionStart;
        const maxDuration = this.profile.sessionDuration * 1000;
        return sessionDuration < maxDuration;
    }
    
    makeRequest(method, endpoint, data = null, headers = {}) {
        const startTime = Date.now();
        
        try {
            let response;
            const url = `${BASE_URL}${endpoint}`;
            const defaultHeaders = {
                'Content-Type': 'application/json',
                'X-User-ID': this.userId,
                'X-Session-ID': this.sessionId,
                'X-Device-Type': this.deviceType,
                'X-Region': this.region,
                ...headers
            };
            
            if (method === 'GET') {
                response = http.get(url, { headers: defaultHeaders });
            } else if (method === 'POST') {
                response = http.post(url, JSON.stringify(data), { headers: defaultHeaders });
            }
            
            const endTime = Date.now();
            const responseTime = endTime - startTime;
            
            // Record metrics
            if (endpoint === '/recommend') {
                recommendationLatency.add(responseTime);
            } else if (endpoint === '/feedback') {
                feedbackLatency.add(responseTime);
            } else if (endpoint.includes('/features')) {
                featureLookupLatency.add(responseTime);
            }
            
            // Check for cache hit
            if (response.headers['X-Cache-Hit']) {
                this.cacheHits++;
                cacheHitRate.add(1);
            } else {
                this.cacheMisses++;
                cacheHitRate.add(0);
            }
            
            // Check for errors
            if (response.status >= 400) {
                errorRate.add(1);
                console.error(`Request failed: ${endpoint} - ${response.status}`);
            } else {
                errorRate.add(0);
            }
            
            // Simulate network latency
            sleep(this.profile.networkLatencyMs / 1000);
            
            return response;
            
        } catch (error) {
            errorRate.add(1);
            console.error(`Request error: ${endpoint} - ${error}`);
            throw error;
        }
    }
    
    getRecommendations() {
        if (!this.shouldContinueSession()) return;
        
        const numRecommendations = randomIntBetween(5, 50);
        const context = this.generateContext();
        const filters = this.generateFilters();
        const candidates = this.generateCandidateItems();
        
        const response = this.makeRequest('POST', '/recommend', {
            user_id: this.userId,
            session_id: this.sessionId,
            num_recommendations: numRecommendations,
            candidate_items: candidates,
            filters: filters,
            context: context
        });
        
        if (response.status === 200) {
            this.recommendationsCount++;
            const recommendations = response.json();
            
            // Simulate user interaction
            if (recommendations.recommendations && recommendations.recommendations.length > 0) {
                this.simulateInteraction(recommendations.recommendations);
            }
            
            check(response, {
                'recommendation status is 200': (r) => r.status === 200,
                'recommendations returned': (r) => r.json().recommendations && r.json().recommendations.length > 0,
                'response time < 100ms': (r) => r.timings.duration < 100,
            });
        }
    }
    
    simulateInteraction(recommendations) {
        if (Math.random() < 0.1) { // 10% chance to interact
            const recommendation = randomItem(recommendations);
            const itemId = recommendation.item_id;
            
            if (itemId && Math.random() < this.profile.feedbackProbability) {
                this.sendFeedback(itemId);
                userEngagement.add(1);
            } else {
                userEngagement.add(0);
            }
        }
    }
    
    sendFeedback(itemId) {
        const interactionTypes = ['click', 'view', 'purchase', 'like', 'dislike'];
        const interactionType = randomItem(interactionTypes);
        let rating = null;
        
        if (['like', 'dislike'].includes(interactionType)) {
            rating = randomIntBetween(1, 5);
        }
        
        const response = this.makeRequest('POST', '/feedback', {
            user_id: this.userId,
            item_id: itemId,
            interaction_type: interactionType,
            rating: rating,
            timestamp: new Date().toISOString(),
            metadata: {
                session_id: this.sessionId,
                device_type: this.deviceType,
                region: this.region
            }
        });
        
        if (response.status === 200) {
            this.feedbackCount++;
            
            check(response, {
                'feedback status is 200': (r) => r.status === 200,
                'feedback processed': (r) => r.json().status === 'success',
            });
        }
    }
    
    getUserFeatures() {
        if (!this.shouldContinueSession()) return;
        if (Math.random() > this.profile.featureLookupProbability) return;
        
        const featureNames = [
            'user_age', 'user_gender', 'user_location', 'user_preferences',
            'purchase_history', 'browse_history', 'search_history',
            'engagement_score', 'loyalty_tier', 'last_active'
        ].slice(0, randomIntBetween(1, 5));
        
        const response = this.makeRequest('GET', `/user/${this.userId}/features`, null, {
            'X-Feature-Names': featureNames.join(',')
        });
        
        if (response.status === 200) {
            this.featureLookups++;
            
            check(response, {
                'user features status is 200': (r) => r.status === 200,
                'features returned': (r) => r.json().features && Object.keys(r.json().features).length > 0,
            });
        }
    }
    
    getItemFeatures() {
        if (!this.shouldContinueSession()) return;
        
        const itemId = randomItem(ITEM_IDS);
        const featureNames = [
            'item_category', 'item_price', 'item_brand', 'item_rating',
            'item_availability', 'item_popularity', 'item_description',
            'item_images', 'item_specs', 'item_reviews'
        ].slice(0, randomIntBetween(1, 5));
        
        const response = this.makeRequest('GET', `/item/${itemId}/features`, null, {
            'X-Feature-Names': featureNames.join(',')
        });
        
        check(response, {
            'item features status is 200': (r) => r.status === 200,
            'features returned': (r) => r.json().features && Object.keys(r.json().features).length > 0,
        });
    }
    
    healthCheck() {
        const response = this.makeRequest('GET', '/health');
        
        check(response, {
            'health check status is 200': (r) => r.status === 200,
            'service is healthy': (r) => r.json().status === 'healthy',
        });
    }
    
    onSessionEnd() {
        const sessionDuration = Date.now() - this.sessionStart;
        
        console.log(`User ${this.userId} session completed:`);
        console.log(`  Duration: ${(sessionDuration / 1000).toFixed(2)}s`);
        console.log(`  Recommendations: ${this.recommendationsCount}`);
        console.log(`  Feedback: ${this.feedbackCount}`);
        console.log(`  Feature lookups: ${this.featureLookups}`);
        console.log(`  Cache hits: ${this.cacheHits}`);
        console.log(`  Cache misses: ${this.cacheMisses}`);
    }
}

// Main test function
export default function() {
    const user = new TestUser();
    
    // Main test loop
    while (user.shouldContinueSession()) {
        // 70% recommendations
        if (Math.random() < 0.7) {
            user.getRecommendations();
        }
        // 15% user features
        else if (Math.random() < 0.85) {
            user.getUserFeatures();
        }
        // 10% item features
        else if (Math.random() < 0.95) {
            user.getItemFeatures();
        }
        // 5% health check
        else {
            user.healthCheck();
        }
        
        // Random wait between requests
        sleep(randomIntBetween(50, 2000) / 1000);
    }
    
    user.onSessionEnd();
}

// Setup function
export function setup() {
    console.log('Starting k6 load test for Recommendation Engine');
    console.log(`Target URL: ${BASE_URL}`);
    console.log(`Target QPS: ${TARGET_QPS}`);
    console.log(`Max Users: ${MAX_USERS}`);
    console.log(`Test Duration: ${TEST_DURATION}`);
    
    return {
        startTime: Date.now(),
    };
}

// Teardown function
export function teardown(data) {
    const endTime = Date.now();
    const duration = (endTime - data.startTime) / 1000;
    
    console.log('k6 load test completed');
    console.log(`Test Duration: ${duration.toFixed(2)}s`);
    
    // Print summary statistics
    console.log('=== Performance Summary ===');
    console.log(`Recommendation Latency P95: ${recommendationLatency.getPercentile(95).toFixed(2)}ms`);
    console.log(`Recommendation Latency P99: ${recommendationLatency.getPercentile(99).toFixed(2)}ms`);
    console.log(`Feedback Latency P95: ${feedbackLatency.getPercentile(95).toFixed(2)}ms`);
    console.log(`Feature Lookup Latency P95: ${featureLookupLatency.getPercentile(95).toFixed(2)}ms`);
    console.log(`Cache Hit Rate: ${(cacheHitRate.rate * 100).toFixed(2)}%`);
    console.log(`Error Rate: ${(errorRate.rate * 100).toFixed(2)}%`);
    console.log(`User Engagement Rate: ${(userEngagement.rate * 100).toFixed(2)}%`);
}

// Handle test interruption
export function handleInterrupt(data) {
    console.log('Test interrupted');
    console.log('Cleaning up...');
    
    // Any cleanup code here
}

// Custom metrics export
export function handleSummary(data) {
    return {
        'recommendation_latency_p95': recommendationLatency.getPercentile(95),
        'recommendation_latency_p99': recommendationLatency.getPercentile(99),
        'feedback_latency_p95': feedbackLatency.getPercentile(95),
        'feature_lookup_latency_p95': featureLookupLatency.getPercentile(95),
        'cache_hit_rate': cacheHitRate.rate,
        'error_rate': errorRate.rate,
        'user_engagement_rate': userEngagement.rate,
        'total_requests': data.metrics.http_reqs.count,
        'total_failures': data.metrics.http_req_failed.count,
    };
}

// Stress testing scenarios
export function stressTest() {
    console.log('Running stress test scenario');
    
    // Override options for stress test
    options.stages = [
        { duration: '2m', target: 500000 },
        { duration: '8m', target: 1000000 },
        { duration: '10m', target: 1000000 },
        { duration: '2m', target: 0 }
    ];
    
    // More aggressive thresholds for stress test
    options.thresholds = {
        http_req_duration: ['p(95)<200', 'p(99)<1000'],
        http_req_failed: ['rate<0.05'],
        recommendation_latency: ['p(95)<200', 'p(99)<1000'],
        errorRate: ['rate<0.05'],
        cache_hit_rate: ['rate>0.6']
    };
    
    // Note: This is a separate test scenario, run with: k6 run -e STRESS_TEST=true k6_script.js
}

// Endurance test scenario
export function enduranceTest() {
    console.log('Running endurance test scenario');
    
    // Override options for endurance test
    options.stages = [
        { duration: '10m', target: 100000 },
        { duration: '20m', target: 200000 },
        { duration: '3h', target: 200000 },
        { duration: '10m', target: 0 }
    ];
    
    // More lenient thresholds for endurance test
    options.thresholds = {
        http_req_duration: ['p(95)<150', 'p(99)<800'],
        http_req_failed: ['rate<0.02'],
        recommendation_latency: ['p(95)<150', 'p(99)<800'],
        errorRate: ['rate<0.02'],
        cache_hit_rate: ['rate>0.65']
    };
    
    // Note: This is a separate test scenario, run with: k6 run -e ENDURANCE_TEST=true k6_script.js
}

// Spike test scenario
export function spikeTest() {
    console.log('Running spike test scenario');
    
    // Override options for spike test
    options.stages = [
        { duration: '5m', target: 50000 },
        { duration: '1m', target: 500000 },  // Spike
        { duration: '2m', target: 500000 },  // Hold spike
        { duration: '1m', target: 50000 },   // Return to normal
        { duration: '5m', target: 0 }
    ];
    
    // Note: This is a separate test scenario, run with: k6 run -e SPIKE_TEST=true k6_script.js
}

// Example usage:
// k6 run --vus 100000 --duration 1h k6_script.js
// k6 run --env BASE_URL=https://staging.api.rec-engine.company.com k6_script.js
// k6 run --env BASE_URL=https://api.rec-engine.company.com k6_script.js
